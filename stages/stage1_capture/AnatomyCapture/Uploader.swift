import Foundation

/// Sends a zipped capture session to the Linux receiver (scripts/upload_server/server.py) with a
/// `URLSession` background upload: the HTTP body IS the raw zip file, which survives the app being
/// backgrounded / pocketed and flaky clinic wifi (auto-retry, resumable). Replaces the
/// AirDrop→Mac→ssh hop. Nothing is sent automatically — the review screen calls `upload(...)`.
///
/// Config lives in UserDefaults (set once in a small Settings sheet, or hardcode for a personal
/// build): `upload_base_url` (e.g. https://184-105-3-239.sslip.io) and `upload_token`.
enum UploadConfig {
    /// Baked build values (Info.plist keys UPLOAD_BASE_URL / UPLOAD_TOKEN) win, so a personal build
    /// needs no Settings entry at all; otherwise fall back to what the Settings sheet stored.
    private static func baked(_ key: String) -> String? {
        (Bundle.main.object(forInfoDictionaryKey: key) as? String).flatMap { $0.isEmpty ? nil : $0 }
    }
    static var baseURL: String { baked("UPLOAD_BASE_URL") ?? UserDefaults.standard.string(forKey: "upload_base_url") ?? "" }
    static var token: String { baked("UPLOAD_TOKEN") ?? UserDefaults.standard.string(forKey: "upload_token") ?? "" }
    static var isConfigured: Bool { !baseURL.isEmpty && !token.isEmpty }
}

final class Uploader: NSObject, URLSessionDelegate, URLSessionTaskDelegate, @unchecked Sendable {
    static let shared = Uploader()

    /// Called when a transfer finishes (any thread) so the UI can confirm/deny — background upload
    /// tasks treat an HTTP 401 (bad PIN/token) as a completed response, not an error, so we inspect
    /// the status code here rather than let a rejected upload look like success.
    var onComplete: (@Sendable (_ sessionID: String, _ ok: Bool, _ message: String) -> Void)?

    private lazy var session: URLSession = {
        let cfg = URLSessionConfiguration.background(withIdentifier: "AnatomyCapture.upload")
        cfg.isDiscretionary = false
        cfg.sessionSendsLaunchEvents = true
        cfg.allowsCellularAccess = true
        return URLSession(configuration: cfg, delegate: self, delegateQueue: nil)
    }()

    /// Kick off a background upload of `zipURL` for session `sessionID`. Returns immediately;
    /// the OS finishes the transfer even if the app is suspended.
    func upload(zipURL: URL, sessionID: String, pin: String = "") -> Result<Void, Error> {
        guard UploadConfig.isConfigured,
              var comps = URLComponents(string: UploadConfig.baseURL) else {
            return .failure(UploadError.notConfigured)
        }
        comps.path = "/upload"
        comps.queryItems = [URLQueryItem(name: "session", value: sessionID)]
        guard let url = comps.url else { return .failure(UploadError.badURL) }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("Bearer \(UploadConfig.token)", forHTTPHeaderField: "Authorization")
        req.setValue("application/zip", forHTTPHeaderField: "Content-Type")
        if !pin.isEmpty { req.setValue(pin, forHTTPHeaderField: "X-Upload-Pin") }   // per-transmit gate

        // fromFile: (not in-memory Data) is required for background uploads.
        let task = session.uploadTask(with: req, fromFile: zipURL)
        task.taskDescription = sessionID
        task.resume()
        return .success(())
    }

    // Surface completion/failure (the UI polls status or shows a toast; kept minimal here).
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let sid = task.taskDescription ?? "?"
        let ok: Bool
        let message: String
        if let error {
            ok = false; message = error.localizedDescription
        } else if let http = task.response as? HTTPURLResponse {
            ok = (200...299).contains(http.statusCode)
            message = ok ? "sent" : (http.statusCode == 401 ? "rejected — wrong PIN or token" : "server error \(http.statusCode)")
        } else {
            ok = false; message = "no response"
        }
        NSLog("[upload] %@ -> ok=%d %@", sid, ok ? 1 : 0, message)
        onComplete?(sid, ok, message)
    }

    enum UploadError: LocalizedError {
        case notConfigured, badURL
        var errorDescription: String? {
            switch self {
            case .notConfigured: return "Upload not configured (set server URL + token in Settings)."
            case .badURL: return "Upload server URL is invalid."
            }
        }
    }
}

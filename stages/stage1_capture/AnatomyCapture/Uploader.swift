import Foundation

/// Sends a zipped capture session to the Linux receiver (scripts/upload_server/server.py) with a
/// `URLSession` background upload: the HTTP body IS the raw zip file, which survives the app being
/// backgrounded / pocketed and flaky clinic wifi (auto-retry, resumable). Replaces the
/// AirDrop→Mac→ssh hop. Nothing is sent automatically — the review screen calls `upload(...)`.
///
/// Config lives in UserDefaults (set once in a small Settings sheet, or hardcode for a personal
/// build): `upload_base_url` (e.g. https://184-105-3-239.sslip.io) and `upload_token`.
enum UploadConfig {
    static var baseURL: String { UserDefaults.standard.string(forKey: "upload_base_url") ?? "" }
    static var token: String { UserDefaults.standard.string(forKey: "upload_token") ?? "" }
    static var isConfigured: Bool { !baseURL.isEmpty && !token.isEmpty }
}

final class Uploader: NSObject, URLSessionDelegate, URLSessionTaskDelegate, @unchecked Sendable {
    static let shared = Uploader()

    private lazy var session: URLSession = {
        let cfg = URLSessionConfiguration.background(withIdentifier: "AnatomyCapture.upload")
        cfg.isDiscretionary = false
        cfg.sessionSendsLaunchEvents = true
        cfg.allowsCellularAccess = true
        return URLSession(configuration: cfg, delegate: self, delegateQueue: nil)
    }()

    /// Kick off a background upload of `zipURL` for session `sessionID`. Returns immediately;
    /// the OS finishes the transfer even if the app is suspended.
    func upload(zipURL: URL, sessionID: String) -> Result<Void, Error> {
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

        // fromFile: (not in-memory Data) is required for background uploads.
        let task = session.uploadTask(with: req, fromFile: zipURL)
        task.taskDescription = sessionID
        task.resume()
        return .success(())
    }

    // Surface completion/failure (the UI polls status or shows a toast; kept minimal here).
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error { NSLog("[upload] %@ failed: %@", task.taskDescription ?? "?", error.localizedDescription) }
        else if let http = task.response as? HTTPURLResponse {
            NSLog("[upload] %@ -> HTTP %d", task.taskDescription ?? "?", http.statusCode)
        }
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

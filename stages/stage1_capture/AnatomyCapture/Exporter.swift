import Foundation

/// Zips a capture folder for AirDrop / "Save to Files" without any dependency,
/// using `NSFileCoordinator`'s `.forUploading` option (which hands back a
/// temporary `.zip` of the directory). The session folder is ALSO always present
/// in the Files app (On My iPhone → AnatomyCapture) via the Info.plist keys, so
/// this is a convenience, not the only path off-device.
enum Exporter {
    static func zip(directory: URL) -> URL? {
        var coordinatorError: NSError?
        var out: URL?
        let sessionName = directory.deletingLastPathComponent().lastPathComponent
        NSFileCoordinator().coordinate(readingItemAt: directory, options: [.forUploading],
                                       error: &coordinatorError) { zippedURL in
            // Unique destination per call: a finalize zip and a transmit re-zip can run close
            // together, and a fixed path would let them race on removeItem/copyItem.
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent("\(sessionName)_\(UUID().uuidString).zip")
            do {
                try FileManager.default.copyItem(at: zippedURL, to: dest)
                out = dest
            } catch {
                out = nil
            }
        }
        return out
    }
}

import SwiftUI

/// AnatomyCapture — Stage 1 of the reconstruction pipeline.
///
/// Records a short patient orbit with the rear LiDAR and writes the exact
/// `capture/` contract (io_contracts/capture_session.md): synchronized color,
/// metric depth, a validity mask, intrinsics, a metric camera path, and
/// timestamps — all from one `ARWorldTrackingConfiguration` session (see
/// IOS_NOTES.md for why the two-stream design is not runnable).
@main
struct AnatomyCaptureApp: App {
    @State private var model = CaptureModel()

    var body: some Scene {
        WindowGroup {
            CaptureView(model: model)
        }
    }
}

import Foundation
import UIKit

/// Which capture framework produced a session.
enum CaptureBackend: String, CaseIterable, Sendable {
    case arkit          // ARKit ARSession: smoothed LiDAR scene-depth + metric camera pose
    case hqDepth        // AVFoundation AVCaptureDepthDataOutput: raw absolute LiDAR depth, NO pose

    var uiLabel: String { self == .arkit ? "ARKit (pose)" : "HQ-Depth (raw)" }
    /// AVFoundation sessions carry no pose; poses.json is omitted and the Linux
    /// pipeline recovers pose via unseeded SfM.
    var providesPose: Bool { self == .arkit }
}

/// Framing orientation the clinician chose. FIDELITY-SAFE: this is a *record + preview*
/// affordance only — the saved color/depth buffers and intrinsics are ALWAYS written in
/// the sensor-native (un-rotated) frame (see IOS_NOTES.md §6, "the #1 pipeline bug"), so
/// the reconstruction is orientation-agnostic. The value is stored purely as metadata and
/// used to orient the on-screen preview/guidance.
enum CaptureOrientation: String, CaseIterable, Sendable {
    case portrait
    case landscape
    var uiLabel: String { self == .portrait ? "Portrait" : "Landscape" }
}

/// User + auto metadata for a capture, written to `capture/metadata.json` and shipped in
/// the zip so the Linux side knows: what was recorded (description), which framework
/// (arkit/hq-depth → whether to run unseeded SfM), framing orientation, and timestamps.
struct CaptureMetadata: Sendable {
    var sessionID: String
    var description: String                 // user-typed, e.g. "left forearm flap, post-debridement"
    var backend: CaptureBackend
    var orientation: CaptureOrientation
    var capturedAt: Date                    // recording start (UTC)
    var finalizedAt: Date                   // finalize time (UTC)
    var frameCount: Int
    var providesPose: Bool
    var depthSource: String                 // human-readable depth provenance
    var deviceModel: String = UIDevice.current.model
    var systemVersion: String = UIDevice.current.systemVersion
    var appVersion: String = (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "?"

    private static let iso: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter(); f.formatOptions = [.withInternetDateTime]; return f
    }()

    /// Plain dictionary for JSONSerialization (matches the other capture JSON files).
    func dictionary() -> [String: Any] {
        [
            "session_id": sessionID,
            "description": description,
            "framework": backend.rawValue,           // "arkit" | "hqDepth"
            "orientation": orientation.rawValue,      // "portrait" | "landscape"
            "captured_at": Self.iso.string(from: capturedAt),
            "finalized_at": Self.iso.string(from: finalizedAt),
            "frame_count": frameCount,
            "provides_pose": providesPose,            // false for hqDepth → pipeline runs unseeded SfM
            "depth_source": depthSource,
            "device_model": deviceModel,
            "ios_version": systemVersion,
            "app_version": appVersion,
        ]
    }
}

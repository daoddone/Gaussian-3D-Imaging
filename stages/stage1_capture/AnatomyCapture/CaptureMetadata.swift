import Foundation

/// Which capture framework produced a session.
enum CaptureBackend: String, CaseIterable, Sendable {
    case arkit          // ARKit ARSession: smoothed LiDAR scene-depth + metric camera pose
    case hqDepth        // AVFoundation AVCaptureDepthDataOutput: raw absolute LiDAR depth, NO pose

    var uiLabel: String { self == .arkit ? "ARKit (pose)" : "HQ-Depth (raw)" }
    /// AVFoundation sessions carry no pose; poses.json is omitted and the Linux
    /// pipeline recovers pose via unseeded SfM.
    var providesPose: Bool { self == .arkit }
}

/// User-selectable capture MODE (the resolution/sensor matrix for the T4 A/B). A mode maps onto
/// one of the two low-level `CaptureBackend`s (which framework owns the camera); UI/preview
/// branches keep reading the derived `CaptureModel.backend`.
///   • arkit1080 — today's default: streamed ~1920x1440 ARKit frames + pose (+ LiDAR depth).
///   • arkit4K   — same ARKit stream/pose, but each KEYFRAME is a 12 MP still via
///                 `ARSession.captureHighResolutionFrame` (the streamed format alone is ~1440p,
///                 identical to arkit1080 — the async still IS the resolution gain).
///   • hqStills  — AVFoundation backend with an `AVCapturePhotoOutput` still (+ photo depth) per
///                 keyframe; falls back to the existing max-res video stream if the photo output
///                 can't coexist with streaming depth (recorded as hq_stills_fallback).
enum CaptureMode: String, CaseIterable, Sendable {
    case arkit1080
    case arkit4K
    case hqStills

    var uiLabel: String {
        switch self {
        case .arkit1080: return "ARKit 1080"
        case .arkit4K:   return "ARKit 4K"
        case .hqStills:  return "HQ Stills"
        }
    }
    /// The low-level camera owner for this mode.
    var backend: CaptureBackend { self == .hqStills ? .hqDepth : .arkit }
}

/// Framing orientation, AUTO-DETECTED from the interface orientation at record time (the UI
/// rotates freely). FIDELITY-SAFE: the saved color/depth buffers and intrinsics are ALWAYS
/// written in the sensor-native (un-rotated) frame (see IOS_NOTES.md §6, "the #1 pipeline bug"),
/// so the reconstruction is orientation-agnostic. The value is descriptive metadata only.
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
    var captureMode: CaptureMode            // the user-selected resolution/sensor mode (T4 A/B)
    var lidarEnabled: Bool                  // false → depth files are all-NaN placeholders (RGB only)
    var orientation: CaptureOrientation
    var capturedAt: Date                    // recording start (UTC)
    var finalizedAt: Date                   // finalize time (UTC)
    var frameCount: Int
    var providesPose: Bool
    var depthSource: String                 // human-readable depth provenance
    // hqStills only: true when AVCapturePhotoOutput could not be configured and the capture fell
    // back to the pre-T4 max-res video stream (so the A/B knows no stills were even attempted).
    var hqStillsFallback: Bool = false
    // Set by the caller on @MainActor (UIDevice.current / Bundle.main are MainActor-isolated).
    var deviceModel: String
    var systemVersion: String
    var appVersion: String

    /// Plain dictionary for JSONSerialization (matches the other capture JSON files).
    func dictionary() -> [String: Any] {
        [
            "session_id": sessionID,
            "description": description,
            "framework": backend.rawValue,           // "arkit" | "hqDepth"
            "capture_mode": captureMode.rawValue,     // "arkit1080" | "arkit4K" | "hqStills"
            "lidar_enabled": lidarEnabled,
            "orientation": orientation.rawValue,      // "portrait" | "landscape"
            "captured_at": capturedAt.ISO8601Format(),
            "finalized_at": finalizedAt.ISO8601Format(),
            "frame_count": frameCount,
            "provides_pose": providesPose,            // false for hqDepth → pipeline runs unseeded SfM
            "depth_source": depthSource,
            "hq_stills_fallback": hqStillsFallback,
            "device_model": deviceModel,
            "ios_version": systemVersion,
            "app_version": appVersion,
        ]
    }
}

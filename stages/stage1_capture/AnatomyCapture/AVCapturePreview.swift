import SwiftUI
import AVFoundation

/// Live preview for the AVFoundation (HQ-Depth) path: an `AVCaptureVideoPreviewLayer` fed by
/// the source's `AVCaptureSession`.
///
/// The preview is DISPLAY ONLY — rotating it here never touches the sensor-native buffers /
/// intrinsics the source writes (IOS_NOTES §6), so it is safe for the fidelity-safe framing.
///
/// FIX (owner-observed "HQ preview is rotated 90° CW"): a plain `AVCaptureVideoPreviewLayer`
/// defaults to the device's *portrait* connection, so in this landscape-locked UI it showed the
/// feed sideways (the ARKit `ARView` path already renders upright). We drive the preview
/// connection's `videoRotationAngle` from an `AVCaptureDevice.RotationCoordinator`
/// (`videoRotationAngleForHorizonLevelPreview`) — the level-horizon angle for how the phone is
/// physically held, computed by the OS rather than a hardcoded guess, and KVO-updated if the
/// device rotates.
struct AVCapturePreview: UIViewRepresentable {
    let source: AVFoundationCaptureSource

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    /// Retains the rotation coordinator + its KVO observation for the view's lifetime.
    final class Coordinator {
        var rotation: AVCaptureDevice.RotationCoordinator?
        var observation: NSKeyValueObservation?
    }
    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.previewLayer.session = source.session
        v.previewLayer.videoGravity = .resizeAspectFill
        source.startPreview()

        // Keep the preview upright regardless of how the phone is held. Display-only: the source
        // still writes sensor-native buffers/intrinsics (IOS_NOTES §6).
        if let device = source.activeDevice {
            let coord = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: v.previewLayer)
            context.coordinator.rotation = coord
            let apply: () -> Void = { [weak layer = v.previewLayer] in
                guard let conn = layer?.connection else { return }
                let angle = coord.videoRotationAngleForHorizonLevelPreview
                if conn.isVideoRotationAngleSupported(angle) { conn.videoRotationAngle = angle }
            }
            apply()
            context.coordinator.observation = coord.observe(
                \.videoRotationAngleForHorizonLevelPreview, options: [.initial, .new]
            ) { _, _ in DispatchQueue.main.async { apply() } }
        }
        return v
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}

    static func dismantleUIView(_ uiView: PreviewView, coordinator: Coordinator) {
        coordinator.observation?.invalidate()
        coordinator.observation = nil
        coordinator.rotation = nil
        uiView.previewLayer.session = nil
    }
}

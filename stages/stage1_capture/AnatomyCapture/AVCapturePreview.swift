import SwiftUI
import AVFoundation

/// Live preview for the AVFoundation (HQ-Depth) path: an `AVCaptureVideoPreviewLayer` fed by
/// the source's `AVCaptureSession`.
///
/// The preview is DISPLAY ONLY — rotating it or tapping to focus never rewrites the sensor-native
/// buffers/intrinsics the source saves (IOS_NOTES §6). It adds two affordances:
///  • Horizon-level rotation via `AVCaptureDevice.RotationCoordinator` (OS-computed angle, so the
///    feed is upright in any interface orientation) — fixes the "HQ preview is 90° CW" report.
///  • Tap-to-focus: a tap converts to a normalized device point and drives the source's focus.
struct AVCapturePreview: UIViewRepresentable {
    let source: AVFoundationCaptureSource

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

        /// Brief yellow reticle at a tap location — UX feedback that tap-to-focus registered.
        func flashFocusIndicator(at point: CGPoint) {
            let size: CGFloat = 72
            let box = CALayer()
            box.frame = CGRect(x: point.x - size / 2, y: point.y - size / 2, width: size, height: size)
            box.borderColor = UIColor.systemYellow.cgColor
            box.borderWidth = 1.5
            box.cornerRadius = 4
            layer.addSublayer(box)
            let shrink = CABasicAnimation(keyPath: "transform.scale")
            shrink.fromValue = 1.35; shrink.toValue = 1.0; shrink.duration = 0.25
            box.add(shrink, forKey: nil)
            let fade = CABasicAnimation(keyPath: "opacity")
            fade.fromValue = 1.0; fade.toValue = 0.0; fade.duration = 0.9
            fade.beginTime = CACurrentMediaTime() + 0.4
            fade.fillMode = .forwards; fade.isRemovedOnCompletion = false
            box.add(fade, forKey: nil)
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.4) { box.removeFromSuperlayer() }
        }
    }

    /// Retains the rotation coordinator + KVO observation and routes taps to the source.
    /// `@MainActor`: every touch point (SwiftUI make/updateUIView, UIGestureRecognizer callbacks,
    /// KVO→main hops, DispatchQueue.main dispatches, PreviewView/CALayer/RotationCoordinator reads)
    /// is main-thread only, so isolating the whole class to MainActor matches its actual runtime
    /// behavior and lets Swift 6.2 reason about the capture in `applyRotationWhenReady`.
    @MainActor
    final class Coordinator: NSObject {
        let source: AVFoundationCaptureSource
        var rotation: AVCaptureDevice.RotationCoordinator?
        var observation: NSKeyValueObservation?
        weak var previewLayer: AVCaptureVideoPreviewLayer?
        init(source: AVFoundationCaptureSource) { self.source = source }

        /// Set the preview connection's rotation to the current horizon-level angle (no-op until
        /// the connection exists, which happens after the source's async configuration completes).
        func applyRotation() {
            guard let conn = previewLayer?.connection, let coord = rotation else { return }
            let angle = coord.videoRotationAngleForHorizonLevelPreview
            if conn.isVideoRotationAngleSupported(angle) { conn.videoRotationAngle = angle }
        }

        /// The preview connection appears only after the session configures on its own queue; poll
        /// briefly so the first upright frame lands even if the device never physically rotates.
        func applyRotationWhenReady(_ attempts: Int) {
            if previewLayer?.connection != nil { applyRotation(); return }
            guard attempts > 0 else { return }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) { [weak self] in
                self?.applyRotationWhenReady(attempts - 1)
            }
        }

        @objc func handleTap(_ g: UITapGestureRecognizer) {
            guard let view = g.view as? PreviewView else { return }
            let layerPoint = g.location(in: view)
            let devicePoint = view.previewLayer.captureDevicePointConverted(fromLayerPoint: layerPoint)
            source.focus(atDevicePoint: devicePoint)
            view.flashFocusIndicator(at: layerPoint)
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(source: source) }

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        context.coordinator.previewLayer = v.previewLayer
        v.previewLayer.session = source.session
        v.previewLayer.videoGravity = .resizeAspectFill
        source.startPreview()

        v.addGestureRecognizer(UITapGestureRecognizer(
            target: context.coordinator, action: #selector(Coordinator.handleTap(_:))))

        // Horizon-level rotation (display-only; buffers stay sensor-native, IOS_NOTES §6).
        if let device = source.displayDevice {
            let coord = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: v.previewLayer)
            context.coordinator.rotation = coord
            context.coordinator.observation = coord.observe(
                \.videoRotationAngleForHorizonLevelPreview, options: [.initial, .new]
            ) { [weak c = context.coordinator] _, _ in
                DispatchQueue.main.async { c?.applyRotation() }
            }
        }
        context.coordinator.applyRotationWhenReady(12)   // ~1.8 s window for the async connection
        return v
    }

    // SwiftUI re-lays-out on interface rotation; re-apply so the preview tracks the new orientation.
    func updateUIView(_ uiView: PreviewView, context: Context) { context.coordinator.applyRotation() }

    static func dismantleUIView(_ uiView: PreviewView, coordinator: Coordinator) {
        coordinator.observation?.invalidate()
        coordinator.observation = nil
        coordinator.rotation = nil
        coordinator.previewLayer = nil
        uiView.previewLayer.session = nil
    }
}

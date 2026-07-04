import SwiftUI
import AVFoundation

/// Live preview for the AVFoundation (HQ-Depth) path: an `AVCaptureVideoPreviewLayer` fed by
/// the source's `AVCaptureSession`. The preview is DISPLAY ONLY — any rotation here does not
/// touch the sensor-native buffers/intrinsics the source writes (IOS_NOTES §6), so it is safe
/// for the fidelity-safe portrait affordance.
struct AVCapturePreview: UIViewRepresentable {
    let source: AVFoundationCaptureSource

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.previewLayer.session = source.session
        v.previewLayer.videoGravity = .resizeAspectFill
        source.startPreview()
        return v
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}

    static func dismantleUIView(_ uiView: PreviewView, coordinator: ()) {
        uiView.previewLayer.session = nil
    }
}

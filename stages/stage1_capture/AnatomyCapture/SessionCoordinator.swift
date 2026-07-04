import Foundation
import ARKit
import CoreImage
import CoreVideo
import CoreGraphics
import simd

/// The `ARSessionDelegate`. Callbacks arrive on the ARSession's background
/// `delegateQueue` (a serial queue), so the whole type is `nonisolated` under
/// Swift 6.2's MainActor-by-default. All mutable state below is confined to that
/// serial queue: the delegate callbacks and the UI-initiated `startWriting` /
/// `stopWriting` calls are all dispatched onto the same queue, so there is no
/// concurrent access. It never retains an `ARFrame` — it copies what it needs
/// and returns immediately.
/// `@unchecked Sendable`: the model hands this coordinator to the delegate queue
/// (via `DispatchQueue.async`) and back through a continuation; all of its
/// mutable state is confined to that single serial queue, so the crossing is
/// safe even though the compiler cannot prove it.
nonisolated final class SessionCoordinator: NSObject, ARSessionDelegate, @unchecked Sendable {

    weak var model: CaptureModel?
    private let ciContext: CIContext
    private let colorSpace: CGColorSpace
    private let confidenceThreshold: UInt8
    var depthMode: String

    // Recording state — delegate-queue confined.
    private var recording = false
    private var writer: FrameWriter?
    private var selector = KeyframeSelector()
    private var index = 0
    private var firstKeptTime: TimeInterval?
    private var trackingNormalThroughout = true
    private var lastPreviewReport: TimeInterval = 0     // throttle for the live preview valid-depth readout
    let cloud = PointCloudAccumulator()                 // world coverage cloud for the post-record inspector

    init(ciContext: CIContext, colorSpace: CGColorSpace, confidenceThreshold: UInt8, depthMode: String) {
        self.ciContext = ciContext
        self.colorSpace = colorSpace
        self.confidenceThreshold = confidenceThreshold
        self.depthMode = depthMode
    }

    /// Begin recording. MUST be dispatched onto the delegate queue by the model.
    func startWriting(writer: FrameWriter) {
        self.writer = writer
        recording = true
        selector.reset()
        index = 0
        firstKeptTime = nil
        trackingNormalThroughout = true
        cloud.reset()
    }

    /// The accumulated world coverage cloud (delegate-queue confined; read at stop on that queue).
    func cloudPoints() -> [SIMD3<Float>] { cloud.points }

    /// Stop recording; no further frames are appended. Returns whether tracking
    /// stayed normal for the whole recording. MUST be dispatched onto the
    /// delegate queue by the model (so it serializes after any in-flight frame).
    func stopWriting() -> Bool {
        recording = false
        let normal = trackingNormalThroughout
        writer = nil
        return normal
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard recording, let writer else {
            if !recording { reportPreviewValid(frame) }   // live framing aid before Record
            return
        }

        guard case .normal = frame.camera.trackingState else {
            trackingNormalThroughout = false
            return
        }
        let time = frame.timestamp

        if selector.isFinished(now: time) {
            recording = false
            Task { @MainActor [weak model] in model?.finishFromBudget() }
            return
        }

        let m = frame.camera.transform
        let rArkit = simd_float3x3(
            SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
            SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
            SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z))

        guard selector.shouldKeep(rotation: rArkit, time: time) else { return }
        guard let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth else { return }
        guard let cg = PixelBufferCopy.colorCGImage(frame.capturedImage, ctx: ciContext, colorSpace: colorSpace) else { return }

        let colorW = CVPixelBufferGetWidth(frame.capturedImage)
        let colorH = CVPixelBufferGetHeight(frame.capturedImage)

        let (depth, mask, dw, dh) = PixelBufferCopy.depthAndMask(
            depth: depthData.depthMap, confidence: depthData.confidenceMap, threshold: confidenceThreshold)

        var validCount = 0
        for v in mask where v == 255 { validCount += 1 }
        let validFrac = Double(validCount) / Double(max(mask.count, 1))

        let K = frame.camera.intrinsics                       // simd_float3x3, at color res
        let Kd: [[Double]] = [
            [Double(K.columns.0.x), 0, Double(K.columns.2.x)],
            [0, Double(K.columns.1.y), Double(K.columns.2.y)],
            [0, 0, 1],
        ]
        let (R, t) = Conventions.openCVCameraToWorld(from: m)

        if firstKeptTime == nil { firstKeptTime = time }
        let relTime = time - (firstKeptTime ?? time)
        index += 1

        writer.append(FramePayload(
            index: index, color: cg, colorW: colorW, colorH: colorH,
            depth: depth, mask: mask, depthW: dw, depthH: dh,
            R: R, t: t, K: Kd, time: relTime, validDepthFraction: validFrac))

        cloud.add(depth: depth, mask: mask, dw: dw, dh: dh, K: Kd,
                  colorW: colorW, colorH: colorH, R: R, t: t)

        let kept = selector.kept
        Task { @MainActor [weak model] in
            model?.note(frameCount: kept, elapsed: relTime, validFraction: validFrac)
        }
    }

    /// Throttled (~2.5 Hz) live valid-depth readout while previewing, so the clinician can frame to
    /// maximize usable depth BEFORE recording. Same valid rule as recording (confidence >= threshold).
    private func reportPreviewValid(_ frame: ARFrame) {
        let t = frame.timestamp
        guard t - lastPreviewReport > 0.4 else { return }
        lastPreviewReport = t
        guard let dd = frame.smoothedSceneDepth ?? frame.sceneDepth else { return }
        let frac = PixelBufferCopy.arkitValidFraction(
            depth: dd.depthMap, confidence: dd.confidenceMap, threshold: confidenceThreshold)
        Task { @MainActor [weak model] in model?.notePreview(validFraction: frac) }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        Task { @MainActor [weak model] in model?.fail("AR session error: \(error.localizedDescription)") }
    }

    func sessionWasInterrupted(_ session: ARSession) {
        Task { @MainActor [weak model] in model?.note(tracking: "session interrupted") }
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        let msg: String
        switch camera.trackingState {
        case .normal: msg = "tracking normal"
        case .notAvailable: msg = "tracking not available"
        case .limited(let reason): msg = "tracking limited (\(reason))"
        }
        Task { @MainActor [weak model] in model?.note(tracking: msg) }
    }
}

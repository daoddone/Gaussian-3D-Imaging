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
    // T4 mode flags, frozen per-recording by startWriting (Sendable copies from the model).
    private var lidarOn = true
    private var highResStills = false                   // arkit4K: 12 MP still per keyframe

    // LiDAR-off placeholder depth dims — ARKit's LiDAR depth resolution, so the on-disk contract
    // keeps its usual [H,W] shape (FrameWriter drops any frame whose depth.count != depthW*depthH).
    private static let placeholderDepthW = 256
    private static let placeholderDepthH = 192

    // High-res still (arkit4K) cross-queue state. captureHighResolutionFrame's completion arrives
    // on an ARKit-internal queue (NOT the delegate queue), so this tiny bit of shared state is
    // lock-guarded (same pattern as cloudLock below). `stillAccepting` closes the writer gate:
    // stopWriting() flips it under the lock BEFORE the model enqueues finalize, so a completion
    // that lands after stop can never append behind finalize (which would orphan frame files
    // with no pose/timestamp entries). `stillInFlight` serializes captures — ARKit errors on
    // overlapping high-res requests; a gate that fires while one is in flight just writes the
    // streamed fallback frame.
    private let stillLock = NSLock()
    private var stillInFlight = false
    private var stillAccepting = false
    private var stillFailureReported = false
    let cloud = PointCloudAccumulator()                 // world coverage cloud (post-record + live overlay)
    // The accumulator lives on the delegate queue; the live overlay reads from the main thread, so we
    // publish a COW snapshot under a lock (the assignment/read is O(1); the buffer is shared, then
    // copy-on-write diverges when the accumulator next mutates — so the reader sees a stable cloud).
    private let cloudLock = NSLock()
    private var displaySnapshot: [SIMD3<Float>] = []

    init(ciContext: CIContext, colorSpace: CGColorSpace, confidenceThreshold: UInt8, depthMode: String) {
        self.ciContext = ciContext
        self.colorSpace = colorSpace
        self.confidenceThreshold = confidenceThreshold
        self.depthMode = depthMode
    }

    /// Begin recording. MUST be dispatched onto the delegate queue by the model.
    /// `lidarEnabled`/`highResStills` are per-recording Sendable copies of the model's mode flags.
    func startWriting(writer: FrameWriter, lidarEnabled: Bool, highResStills: Bool) {
        self.writer = writer
        recording = true
        lidarOn = lidarEnabled
        self.highResStills = highResStills
        selector = KeyframeSelector()          // re-read CaptureTuning (cap/budget) for this recording
        index = 0
        firstKeptTime = nil
        trackingNormalThroughout = true
        cloud.reset()
        publishCloud()
        stillLock.lock()
        stillInFlight = false
        stillAccepting = true
        stillFailureReported = false
        stillLock.unlock()
    }

    /// The accumulated world coverage cloud (delegate-queue confined; read at stop on that queue).
    func cloudPoints() -> [SIMD3<Float>] { cloud.points }

    /// Publish the current cloud for the live overlay (call on the delegate queue). COW → O(1).
    private func publishCloud() { cloudLock.lock(); displaySnapshot = cloud.points; cloudLock.unlock() }

    /// Thread-safe snapshot of the growing cloud for the live overlay (safe to call from any thread).
    func displayCloudSnapshot() -> [SIMD3<Float>] { cloudLock.lock(); defer { cloudLock.unlock() }; return displaySnapshot }

    /// Stop recording; no further frames are appended. Returns whether tracking
    /// stayed normal for the whole recording. MUST be dispatched onto the
    /// delegate queue by the model (so it serializes after any in-flight frame).
    /// Closing `stillAccepting` under the lock here happens-before the model enqueues
    /// finalize, so a late high-res-still completion is dropped instead of appending
    /// behind finalize.
    func stopWriting() -> Bool {
        recording = false
        stillLock.lock()
        stillAccepting = false
        stillLock.unlock()
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

        // Depth: real LiDAR (confidence-masked) when enabled. With the LiDAR toggle OFF the
        // configuration carries no depth semantics, so synthesize the placeholder the contract
        // shape expects: all-NaN depth sized EXACTLY depthW*depthH (FrameWriter drops any frame
        // whose depth.count mismatches) + all-255 mask. RGB + VIO pose + K are still real.
        let depth: [Float], mask: [UInt8], dw: Int, dh: Int
        if lidarOn {
            guard let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth else { return }
            let d = PixelBufferCopy.depthAndMask(
                depth: depthData.depthMap, confidence: depthData.confidenceMap, threshold: confidenceThreshold)
            depth = d.depth; mask = d.mask; dw = d.w; dh = d.h
        } else {
            dw = Self.placeholderDepthW; dh = Self.placeholderDepthH
            depth = [Float](repeating: .nan, count: dw * dh)
            mask = [UInt8](repeating: 255, count: dw * dh)
        }
        guard let cg = PixelBufferCopy.colorCGImage(frame.capturedImage, ctx: ciContext, colorSpace: colorSpace) else { return }

        let colorW = CVPixelBufferGetWidth(frame.capturedImage)
        let colorH = CVPixelBufferGetHeight(frame.capturedImage)

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

        let payload = FramePayload(
            index: index, color: cg, colorW: colorW, colorH: colorH,
            depth: depth, mask: mask, depthW: dw, depthH: dh,
            R: R, t: t, K: Kd, time: relTime, validDepthFraction: validFrac)

        if highResStills {
            // arkit4K: the 12 MP still is the deliverable; the streamed payload is only the
            // fallback if the still can't be produced (busy/error).
            captureStillAndAppend(session: session, fallback: payload, writer: writer, lidarOn: lidarOn)
        } else {
            writer.append(payload)
        }

        if lidarOn {                                       // no depth with LiDAR off -> no cloud
            cloud.add(depth: depth, mask: mask, dw: dw, dh: dh, K: Kd,
                      colorW: colorW, colorH: colorH, R: R, t: t)
            publishCloud()                                 // refresh the live overlay snapshot
        }

        let kept = selector.kept
        Task { @MainActor [weak model] in
            model?.note(frameCount: kept, elapsed: relTime, validFraction: validFrac)
        }
    }

    // MARK: - arkit4K 12 MP still (the T4 resolution gain)

    /// Pull the 12 MP still for this keyframe and write IT (with its own pose/K/sceneDepth)
    /// instead of the streamed frame. The streamed "high-res" videoFormat is still ~1920x1440 —
    /// identical to arkit1080 — so this async still IS the entire resolution gain; without it the
    /// arkit1080/arkit4K A/B would compare two byte-identical configs.
    ///
    /// Concurrency: called on the delegate queue; the completion arrives on an ARKit-INTERNAL
    /// queue. The completion touches only: value copies made at gate time (`fallback`, `lidarOn`),
    /// `self`'s immutable lets (CIContext is documented thread-safe), the stillLock-guarded flags,
    /// and `FrameWriter.append` (which only enqueues onto the writer's private serial IO queue).
    /// Delegate-queue state (index/selector/cloud/firstKeptTime) is NEVER read from the completion —
    /// the frame's index and relative time were frozen into `fallback` on the delegate queue.
    /// The ARFrame is copied out inside the completion and never escapes it.
    private func captureStillAndAppend(session: ARSession, fallback: FramePayload,
                                       writer: FrameWriter, lidarOn: Bool) {
        stillLock.lock()
        let busy = stillInFlight
        if !busy { stillInFlight = true }
        stillLock.unlock()
        if busy {                       // one still at a time (ARKit errors on overlapping requests)
            writer.append(fallback)     // keep this keyframe at stream res rather than dropping it
            return
        }
        session.captureHighResolutionFrame { [self] hiFrame, error in
            guard let hiFrame, error == nil,
                  let cg = PixelBufferCopy.colorCGImage(hiFrame.capturedImage,
                                                        ctx: ciContext, colorSpace: colorSpace) else {
                noteStillFailureOnce(error)
                finishStill(appending: fallback, to: writer)
                return
            }
            let hiW = CVPixelBufferGetWidth(hiFrame.capturedImage)
            let hiH = CVPixelBufferGetHeight(hiFrame.capturedImage)

            // Prefer the still's OWN sceneDepth (LiDAR keeps streaming during the capture —
            // verified in the T4 spec). If it's missing, reuse the streamed frame's depth from
            // `fallback` (milliseconds apart, same 256x192 contract). LiDAR off -> the fallback
            // already carries the all-NaN placeholder.
            var depth = fallback.depth, mask = fallback.mask
            var dw = fallback.depthW, dh = fallback.depthH
            var validFrac = fallback.validDepthFraction
            if lidarOn, let dd = hiFrame.smoothedSceneDepth ?? hiFrame.sceneDepth {
                let d = PixelBufferCopy.depthAndMask(depth: dd.depthMap, confidence: dd.confidenceMap,
                                                     threshold: confidenceThreshold)
                depth = d.depth; mask = d.mask; dw = d.w; dh = d.h
                var valid = 0
                for v in mask where v == 255 { valid += 1 }
                validFrac = Double(valid) / Double(max(mask.count, 1))
            }

            // K: ARKit reports camera.intrinsics at camera.imageResolution (the STREAM res, not
            // the 12 MP still), so scale to the still's pixel size — same ratio discipline as the
            // AVFoundation path. If an SDK ever reports intrinsics at the still's own resolution,
            // the ratios collapse to 1 and this is a no-op.
            let K = hiFrame.camera.intrinsics
            let ref = hiFrame.camera.imageResolution
            let sx = ref.width  > 0 ? Double(hiW) / Double(ref.width)  : 1
            let sy = ref.height > 0 ? Double(hiH) / Double(ref.height) : 1
            let Kd: [[Double]] = [
                [Double(K.columns.0.x) * sx, 0, Double(K.columns.2.x) * sx],
                [0, Double(K.columns.1.y) * sy, Double(K.columns.2.y) * sy],
                [0, 0, 1],
            ]
            let (R, t) = Conventions.openCVCameraToWorld(from: hiFrame.camera.transform)

            finishStill(appending: FramePayload(
                index: fallback.index, color: cg, colorW: hiW, colorH: hiH,
                depth: depth, mask: mask, depthW: dw, depthH: dh,
                R: R, t: t, K: Kd, time: fallback.time, validDepthFraction: validFrac), to: writer)
        }
    }

    /// Completion-side sink: clear the in-flight flag and append ONLY while the recording still
    /// accepts frames. The append happens under stillLock, so it is enqueued on the writer's IO
    /// queue strictly before a concurrent stopWriting() (same lock) can return and let the model
    /// enqueue finalize — no orphan frame files behind finalize.
    private func finishStill(appending payload: FramePayload, to writer: FrameWriter) {
        stillLock.lock()
        defer { stillLock.unlock() }
        stillInFlight = false
        guard stillAccepting else { return }
        writer.append(payload)
    }

    /// Surface the FIRST still failure of a recording in the tracking bar (after that, frames
    /// silently fall back to the streamed ~1440p image — the capture keeps its cadence either way,
    /// but the owner should know the 4K A/B leg degraded; the PNG dimensions are the ground truth).
    private func noteStillFailureOnce(_ error: Error?) {
        stillLock.lock()
        let first = !stillFailureReported
        stillFailureReported = true
        stillLock.unlock()
        guard first else { return }
        let msg = error?.localizedDescription ?? "no frame returned"
        Task { @MainActor [weak model] in
            model?.note(tracking: "4K still failed (\(msg)) — saving stream-res frames")
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

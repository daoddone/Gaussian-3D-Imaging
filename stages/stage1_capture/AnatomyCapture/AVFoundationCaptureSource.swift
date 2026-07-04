import Foundation
import AVFoundation
import CoreImage
import CoreVideo
import CoreGraphics
import simd

/// Path B: the AVFoundation "high-quality depth" capture source. Drives an
/// `AVCaptureSession` on `.builtInLiDARDepthCamera` with a hand-picked highest-resolution
/// depth format (raw, unfiltered, `.absolute` metric) synchronized to high-res color. It
/// produces the SAME `FramePayload` the ARKit path does — MINUS the pose (R/t = nil), which
/// AVFoundation streaming does not provide; the Linux pipeline recovers pose via unseeded SfM.
///
/// Mirrors `SessionCoordinator`'s discipline: all mutable capture state is confined to the
/// synchronizer's serial `dataQueue`; buffers are copied out immediately and never retained.
/// Preview: expose `session` to an `AVCaptureVideoPreviewLayer` (display-only rotation is
/// fine and does NOT affect the sensor-native buffers/intrinsics we save — see IOS_NOTES §6).
///
/// ON-DEVICE VERIFY (cannot be checked off-hardware): (1) a LiDAR format with non-empty
/// `supportedDepthDataFormats` is found; (2) `depthDataAccuracy == .absolute`; (3)
/// `cameraCalibrationData` is non-nil; (4) run the §7 orientation self-test on a near-planar
/// frame to confirm K↔image consistency before trusting output.
nonisolated final class AVFoundationCaptureSource: NSObject,
        AVCaptureDataOutputSynchronizerDelegate, @unchecked Sendable {

    weak var model: CaptureModel?
    let session = AVCaptureSession()

    /// The LiDAR device backing the session, exposed so the preview can build a
    /// `RotationCoordinator` for a horizon-level display (display-only; see `AVCapturePreview`).
    private(set) var activeDevice: AVCaptureDevice?

    /// Device to build the preview's rotation coordinator with. `AVCaptureDevice.default` returns
    /// the same shared LiDAR singleton the session uses, so we look it up fresh (main-thread safe,
    /// available before the async `configureIfNeeded` sets `activeDevice`) rather than read the
    /// sessionQueue-confined `activeDevice` across threads.
    var displayDevice: AVCaptureDevice? {
        AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: .back)
    }

    private let ciContext: CIContext
    private let colorSpace: CGColorSpace
    private let sessionQueue = DispatchQueue(label: "avf.session", qos: .userInitiated)
    private let dataQueue = DispatchQueue(label: "avf.data", qos: .userInitiated)

    private let videoOut = AVCaptureVideoDataOutput()
    private let depthOut = AVCaptureDepthDataOutput()
    private var synchronizer: AVCaptureDataOutputSynchronizer?
    private var configured = false

    // dataQueue-confined recording state (same shape as SessionCoordinator).
    private var recording = false
    private var writer: FrameWriter?
    private var selector = KeyframeSelector()
    private var index = 0
    private var firstKeptTime: TimeInterval?
    private var lastPreviewReport: TimeInterval = 0     // throttle for the live preview valid-depth readout
    private let identityR = matrix_identity_float3x3   // no live pose -> selector uses time-stride only
    private let cloud = PointCloudAccumulator()         // per-frame single-view cloud (no pose to fuse)
    private var bestCloud: [SIMD3<Float>] = []          // densest kept frame -> the post-record inspector

    init(ciContext: CIContext, colorSpace: CGColorSpace) {
        self.ciContext = ciContext
        self.colorSpace = colorSpace
        super.init()
    }

    /// LiDAR + depth availability for Path B on this device.
    static func isSupported() -> Bool {
        AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: .back) != nil
    }

    // MARK: - configuration + lifecycle

    /// Configure inputs/outputs + pick the highest-res depth format. Returns false (via model.fail)
    /// if the LiDAR device / a depth-capable format is unavailable.
    private var lastConfigError = ""       // set by configureIfNeeded; reported by startPreview only after retries

    private func configureIfNeeded() -> Bool {
        if configured { return true }
        guard let device = AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: .back) else {
            lastConfigError = "no builtInLiDARDepthCamera on this device"; return false
        }
        activeDevice = device
        session.beginConfiguration()
        var committed = false
        // On any failure remove partially-added I/O so a retry starts clean — a leftover
        // videoOut/depthOut makes canAddOutput() return false ("cannot add video output") forever.
        defer {
            if !committed {
                for o in session.outputs { session.removeOutput(o) }
                for i in session.inputs { session.removeInput(i) }
                session.commitConfiguration()
            }
        }
        session.sessionPreset = .inputPriority        // MUST: else the session overrides activeFormat

        do {
            let input = try AVCaptureDeviceInput(device: device)
            guard session.canAddInput(input) else { lastConfigError = "cannot add camera input"; return false }
            session.addInput(input)
        } catch {
            lastConfigError = "camera input error: \(error.localizedDescription)"; return false
        }

        guard session.canAddOutput(videoOut) else { lastConfigError = "cannot add video output"; return false }
        videoOut.alwaysDiscardsLateVideoFrames = true
        session.addOutput(videoOut)

        guard session.canAddOutput(depthOut) else { lastConfigError = "cannot add depth output"; return false }
        depthOut.isFilteringEnabled = false           // raw depth; holes arrive as NaN (finite-mask them)
        depthOut.alwaysDiscardsLateDepthData = true
        session.addOutput(depthOut)

        // Highest-res color format that also carries depth, then its highest-res depth format.
        guard let colorFormat = device.formats.filter({ !$0.supportedDepthDataFormats.isEmpty && !$0.isVideoBinned })
            .max(by: { a, b in
                let da = CMVideoFormatDescriptionGetDimensions(a.formatDescription)
                let db = CMVideoFormatDescriptionGetDimensions(b.formatDescription)
                return Int(da.width) * Int(da.height) < Int(db.width) * Int(db.height)
            }) else {
            lastConfigError = "no color format supports depth on this device"; return false
        }
        guard let depthFormat = colorFormat.supportedDepthDataFormats.filter({
            CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat16
        }).max(by: { a, b in
            let da = CMVideoFormatDescriptionGetDimensions(a.formatDescription)
            let db = CMVideoFormatDescriptionGetDimensions(b.formatDescription)
            return Int(da.width) * Int(da.height) < Int(db.width) * Int(db.height)
        }) else {
            lastConfigError = "no DepthFloat16 depth format available"; return false
        }
        do {
            try device.lockForConfiguration()
            device.activeFormat = colorFormat
            device.activeDepthDataFormat = depthFormat
            // Continuous autofocus by default: without this the LiDAR device can sit at a fixed
            // far focus (the "won't focus closer than ~12 in" symptom). Per-frame K captures the
            // resulting focus-breathing so metric accuracy is preserved (Stage 3 uses K_per_frame).
            if device.isFocusModeSupported(.continuousAutoFocus) { device.focusMode = .continuousAutoFocus }
            if device.isExposureModeSupported(.continuousAutoExposure) { device.exposureMode = .continuousAutoExposure }
            device.unlockForConfiguration()
        } catch {
            lastConfigError = "could not set active formats: \(error.localizedDescription)"; return false
        }

        let sync = AVCaptureDataOutputSynchronizer(dataOutputs: [depthOut, videoOut])
        sync.setDelegate(self, queue: dataQueue)
        synchronizer = sync
        session.commitConfiguration()
        committed = true
        configured = true
        return true
    }

    /// Start the live preview session (call when the toggle selects HQ-Depth, or to recover after a
    /// failure). Retries: on a backend swap the outgoing ARSession may not have released the LiDAR
    /// camera yet, so lockForConfiguration/addInput can throw transiently — we back off and retry,
    /// and only surface a failure if all attempts fail (no per-attempt .failed flicker).
    func startPreview() {
        sessionQueue.async { [self] in
            if session.isRunning { return }
            var ok = false
            for attempt in 0..<4 {
                if configureIfNeeded() { ok = true; break }
                if attempt < 3 { Thread.sleep(forTimeInterval: 0.2) }   // let a just-released camera settle
            }
            guard ok else { reportFail(lastConfigError); return }
            if !session.isRunning { session.startRunning() }            // synchronous; hence off the main queue
            // Replace the stuck "starting…" readout (ARKit posts tracking states; HQ has none).
            Task { @MainActor [weak model] in model?.note(tracking: "HQ-Depth ready — tap to focus") }
        }
    }

    func stopPreview() {
        sessionQueue.async { [self] in
            if session.isRunning { session.stopRunning() }
        }
    }

    /// Synchronous stop for a backend swap: the incoming ARSession must NOT start until this
    /// AVCaptureSession has actually released the rear camera (else they fight over it).
    func stopPreviewAndWait() {
        sessionQueue.sync {
            if session.isRunning { session.stopRunning() }
        }
    }

    // MARK: - focus control (HQ path)

    /// Continuous autofocus (keeps refocusing as working distance changes — per-frame K tracks
    /// the drift) vs locked (pins the current lens position for a stable, sharp static close-up).
    private(set) var focusLocked = false

    func setFocusLocked(_ locked: Bool) {
        sessionQueue.async { [self] in
            guard let device = activeDevice else { return }
            do {
                try device.lockForConfiguration()
                if locked {
                    if device.isFocusModeSupported(.locked) { device.focusMode = .locked }
                } else if device.isFocusModeSupported(.continuousAutoFocus) {
                    device.focusMode = .continuousAutoFocus
                }
                device.unlockForConfiguration()
                focusLocked = locked
            } catch {
                reportFail("focus mode: \(error.localizedDescription)")
            }
        }
    }

    /// Tap-to-focus at a normalized device point (0..1, sensor space; the preview converts the
    /// on-screen tap via `captureDevicePointConverted`). In auto mode this re-centres continuous
    /// AF on the point; in locked mode it does a one-shot focus there and then holds. Exposure
    /// follows the same point so the region of interest is both sharp and well-exposed.
    func focus(atDevicePoint p: CGPoint) {
        sessionQueue.async { [self] in
            guard let device = activeDevice else { return }
            do {
                try device.lockForConfiguration()
                if device.isFocusPointOfInterestSupported { device.focusPointOfInterest = p }
                let mode: AVCaptureDevice.FocusMode = focusLocked ? .autoFocus : .continuousAutoFocus
                if device.isFocusModeSupported(mode) { device.focusMode = mode }
                if device.isExposurePointOfInterestSupported { device.exposurePointOfInterest = p }
                if device.isExposureModeSupported(.continuousAutoExposure) {
                    device.exposureMode = .continuousAutoExposure
                }
                device.unlockForConfiguration()
            } catch {
                reportFail("tap focus: \(error.localizedDescription)")
            }
        }
    }

    func startWriting(writer: FrameWriter) {
        dataQueue.async { [self] in
            self.writer = writer
            recording = true
            selector.reset()
            index = 0
            firstKeptTime = nil
            cloud.reset()
            bestCloud = []
        }
    }

    /// The densest single-view coverage cloud (HQ has no pose to fuse frames). dataQueue-confined.
    func cloudPoints() -> [SIMD3<Float>] { dataQueue.sync { bestCloud } }

    func stopWriting() {
        dataQueue.sync { recording = false; writer = nil }
    }

    // MARK: - AVCaptureDataOutputSynchronizerDelegate (dataQueue)

    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput collection: AVCaptureSynchronizedDataCollection) {
        guard let syncedDepth = collection.synchronizedData(for: depthOut) as? AVCaptureSynchronizedDepthData,
              let syncedVideo = collection.synchronizedData(for: videoOut) as? AVCaptureSynchronizedSampleBufferData,
              !syncedDepth.depthDataWasDropped, !syncedVideo.sampleBufferWasDropped else { return }
        guard recording, let writer else {
            reportPreviewValid(syncedDepth.depthData, time: syncedVideo.timestamp.seconds)   // live framing aid
            return
        }

        let time = syncedVideo.timestamp.seconds
        if selector.isFinished(now: time) {
            recording = false
            Task { @MainActor [weak model] in model?.finishFromBudget() }
            return
        }
        // No live pose -> identity rotation: the angular gate never fires, so KeyframeSelector
        // keeps frames purely on its uniform time-stride fallback (~0.417 s).
        guard selector.shouldKeep(rotation: identityR, time: time) else { return }

        var depthData = syncedDepth.depthData
        if depthData.depthDataType != kCVPixelFormatType_DepthFloat32 {
            depthData = depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        }
        guard let calib = depthData.cameraCalibrationData else { return }  // no intrinsics -> drop

        guard let colorPB = CMSampleBufferGetImageBuffer(syncedVideo.sampleBuffer),
              let cg = PixelBufferCopy.colorCGImage(colorPB, ctx: ciContext, colorSpace: colorSpace) else { return }
        let colorW = CVPixelBufferGetWidth(colorPB)
        let colorH = CVPixelBufferGetHeight(colorPB)

        let (depth, mask, dw, dh) = PixelBufferCopy.depthFloat32AndFiniteMask(depthData.depthDataMap)
        var validCount = 0
        for v in mask where v == 255 { validCount += 1 }
        let validFrac = Double(validCount) / Double(max(mask.count, 1))

        // K is expressed at intrinsicMatrixReferenceDimensions; scale to the COLOR buffer res
        // (payload K applies to color, per the capture contract).
        let m = calib.intrinsicMatrix
        let ref = calib.intrinsicMatrixReferenceDimensions
        let sx = Double(colorW) / Double(ref.width)
        let sy = Double(colorH) / Double(ref.height)
        let K: [[Double]] = [
            [Double(m.columns.0.x) * sx, 0,                            Double(m.columns.2.x) * sx],
            [0,                          Double(m.columns.1.y) * sy,   Double(m.columns.2.y) * sy],
            [0,                          0,                            1],
        ]

        if firstKeptTime == nil { firstKeptTime = time }
        let relTime = time - (firstKeptTime ?? time)
        index += 1

        writer.append(FramePayload(
            index: index, color: cg, colorW: colorW, colorH: colorH,
            depth: depth, mask: mask, depthW: dw, depthH: dh,
            R: nil, t: nil,                            // AVFoundation: no pose
            K: K, time: relTime, validDepthFraction: validFrac))

        // HQ has no pose -> can't fuse frames; keep the DENSEST single frame (best coverage view).
        // Fine sampling (step 2) + a tight depth cutoff, since it's just one frame of the subject.
        cloud.reset()
        cloud.add(depth: depth, mask: mask, dw: dw, dh: dh, K: K,
                  colorW: colorW, colorH: colorH,
                  R: PointCloudAccumulator.identityR, t: PointCloudAccumulator.zeroT,
                  step: 2, maxDepth: 1.5)
        if cloud.points.count > bestCloud.count { bestCloud = cloud.points }

        let kept = selector.kept
        Task { @MainActor [weak model] in
            model?.note(frameCount: kept, elapsed: relTime, validFraction: validFrac)
        }
    }

    /// Throttled (~2.5 Hz) live valid-depth readout while previewing (dataQueue). Converts to
    /// Float32 like the recording path and reports the finite fraction, so the clinician can frame
    /// to raise valid-depth before recording (HQ raw LiDAR holes are NaN → low % when too close).
    private func reportPreviewValid(_ depthData: AVDepthData, time: TimeInterval) {
        guard time - lastPreviewReport > 0.4 else { return }
        lastPreviewReport = time
        var dd = depthData
        if dd.depthDataType != kCVPixelFormatType_DepthFloat32 {
            dd = dd.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        }
        let frac = PixelBufferCopy.finiteFraction(depthFloat32: dd.depthDataMap)
        Task { @MainActor [weak model] in model?.notePreview(validFraction: frac) }
    }

    private func reportFail(_ msg: String) {
        // NOTE: does NOT commitConfiguration — configureIfNeeded's `defer` balances beginConfiguration
        // on every failure path (and this is also called before begin, e.g. no-device).
        Task { @MainActor [weak model] in model?.fail("HQ-Depth: \(msg)") }
    }
}

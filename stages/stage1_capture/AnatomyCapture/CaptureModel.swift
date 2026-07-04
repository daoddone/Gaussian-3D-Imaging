import Foundation
import ARKit
import CoreImage
import CoreGraphics
import Metal
import SceneKit
import UIKit
import Observation

/// Owns the capture lifecycle and the UI state. `@MainActor` + `@Observable`.
///
/// TWO capture backends behind a runtime toggle (only switchable when not recording):
///   • ARKit (default): ARSession owned by the RealityKit ARView, handed here via `bind(session:)`;
///     gives smoothed LiDAR depth + metric camera pose.
///   • HQ-Depth: `AVFoundationCaptureSource` (owns its own AVCaptureSession); raw absolute LiDAR
///     depth + high-res color, NO pose (pipeline recovers pose via unseeded SfM).
/// Both feed the same `FrameWriter`. Only one may own the rear camera at a time, so switching
/// backends pauses/stops the other (see `setBackend`).
@MainActor
@Observable
final class CaptureModel {

    enum Phase: Equatable {
        case idle
        case previewing
        case recording
        case finalizing
        case finished(URL)
        case failed(String)
    }

    var phase: Phase = .idle
    var frameCount: Int = 0
    var elapsed: TimeInterval = 0
    var validDepthFraction: Double = 0
    var trackingMessage: String = ""
    var exportURL: URL?

    // Toggles + metadata (bindable from the UI; only mutate when not recording).
    var backend: CaptureBackend = .arkit
    var orientation: CaptureOrientation = .portrait
    var captureDescription: String = ""
    var uploadMessage: String = ""

    // Live coverage-mesh overlay on/off (ARKit only). Display-only — toggling it never affects
    // the saved capture; off = plain camera video, on = LiDAR scene-reconstruction mesh drawn over it.
    var showOverlay: Bool = true

    // Focus mode for the HQ-Depth path (ARKit manages its own autofocus). false = continuous
    // autofocus (default; keeps the subject sharp as distance changes, per-frame K tracks the
    // drift); true = locked lens for a stable static close-up. Tap-to-focus is wired in the preview.
    var focusLocked: Bool = false {
        didSet { if backend == .hqDepth { avSource.setFocusLocked(focusLocked) } }
    }

    // Post-record 3D coverage inspector (ARKit only; nil for HQ-Depth). Feedback, not saved.
    var meshNode: SCNNode?
    var showInspector = false

    let coordinator: SessionCoordinator
    private let ciContext: CIContext
    private let colorSpace: CGColorSpace

    private var session: ARSession?
    private let delegateQueue = DispatchQueue(label: "ar.delegate", qos: .userInitiated)
    private var writer: FrameWriter?
    private var sessionID: String = ""
    private var recordStartedAt = Date()
    private var safetyStop: Task<Void, Never>?

    let budgetSeconds: TimeInterval = 20

    /// Path B source (lazily created; owns its own AVCaptureSession).
    @ObservationIgnored
    lazy var avSource: AVFoundationCaptureSource = {
        let s = AVFoundationCaptureSource(ciContext: ciContext, colorSpace: colorSpace)
        s.model = self
        return s
    }()

    init() {
        let context: CIContext = {
            if let device = MTLCreateSystemDefaultDevice() {
                return CIContext(mtlDevice: device)
            }
            return CIContext()
        }()
        let srgb = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        self.ciContext = context
        self.colorSpace = srgb
        coordinator = SessionCoordinator(ciContext: context, colorSpace: srgb,
                                         confidenceThreshold: 1, depthMode: "smoothedSceneDepth")
        coordinator.model = self
    }

    /// ARKit LiDAR/depth availability.
    static func isSupported() -> Bool {
        ARWorldTrackingConfiguration.isSupported &&
        (ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) ||
         ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth))
    }

    /// Called by the ARView preview when the ARSession exists (or is recreated after a backend
    /// swap): pause any prior session, adopt + configure this one, start live preview.
    func bind(session: ARSession) {
        if self.session === session { return }
        self.session?.pause()
        self.session = session
        session.delegate = coordinator
        session.delegateQueue = delegateQueue
        runConfiguration()
        if case .idle = phase { phase = .previewing }
    }

    private func runConfiguration() {
        let cfg = ARWorldTrackingConfiguration()
        cfg.worldAlignment = .gravity
        cfg.isAutoFocusEnabled = true            // explicit: ARKit autofocuses; per-frame K tracks it
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            cfg.frameSemantics = [.smoothedSceneDepth]
            coordinator.depthMode = "smoothedSceneDepth"
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            cfg.frameSemantics = [.sceneDepth]
            coordinator.depthMode = "sceneDepth"
        }
        // Live coverage overlay: ARKit fuses LiDAR into a world mesh on-device (free). The
        // ARView renders it (showSceneUnderstanding) so the clinician sees covered vs missing
        // regions live and can dwell on thin areas (wound bed / medial arm). Feedback only —
        // NOT written to the capture. Also snapshotted post-record for the 3D inspector.
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            cfg.sceneReconstruction = .mesh
        }
        session?.run(cfg, options: [.resetTracking, .removeExistingAnchors])
    }

    /// The live LiDAR scene-reconstruction mesh anchors (ARKit only), for the live overlay +
    /// the post-record inspector. Empty for the HQ-Depth backend (AVFoundation has no meshing).
    func currentMeshAnchors() -> [ARMeshAnchor] {
        (session?.currentFrame?.anchors ?? []).compactMap { $0 as? ARMeshAnchor }
    }

    /// Switch capture framework (allowed only in idle/previewing). Tears down the outgoing
    /// backend's session so only one owns the camera; the SwiftUI preview swap starts the new one.
    func setBackend(_ b: CaptureBackend) {
        guard b != backend, phase == .idle || phase == .previewing else { return }
        // Tear the outgoing session down BEFORE flipping backend, so the incoming session doesn't
        // start while the other still owns the rear camera (black/frozen preview, interruption).
        switch backend {
        case .arkit:    session?.pause()
        case .hqDepth:  avSource.stopPreviewAndWait()    // synchronous: camera released before ARKit runs
        }
        backend = b
        trackingMessage = ""
        if b == .hqDepth { focusLocked = false }     // fresh HQ config comes up in continuous AF
    }

    // MARK: - recording

    func startRecording() {
        guard phase == .previewing || phase == .idle else { return }
        orientation = currentInterfaceOrientation()      // record how the phone is actually held
        do {
            sessionID = Self.makeSessionID()
            let dir = try Self.captureDirectory(for: sessionID)
            let w = try FrameWriter(captureDir: dir)
            writer = w
            frameCount = 0; elapsed = 0; validDepthFraction = 0
            recordStartedAt = Date()
            phase = .recording
            switch backend {
            case .arkit:   delegateQueue.async { [coordinator] in coordinator.startWriting(writer: w) }
            case .hqDepth: avSource.startWriting(writer: w)
            }
            safetyStop = Task { [weak self] in
                try? await Task.sleep(nanoseconds: UInt64((self?.budgetSeconds ?? 20) + 1) * 1_000_000_000)
                if self?.phase == .recording { await self?.stopRecording() }
            }
        } catch {
            phase = .failed("could not start capture: \(error.localizedDescription)")
        }
    }

    /// Called by either source (via MainActor hop) when the budget/cap is hit.
    func finishFromBudget() { Task { await stopRecording() } }

    func stopRecording() async {
        guard phase == .recording else { return }
        safetyStop?.cancel()
        phase = .finalizing
        guard let w = writer else { phase = .failed("no writer"); return }

        // Stop appends (serialized after any in-flight frame), then finalize (runs after appends).
        var trackingNormal: Bool? = nil
        switch backend {
        case .arkit:
            trackingNormal = await withCheckedContinuation { cont in
                delegateQueue.async { [coordinator] in cont.resume(returning: coordinator.stopWriting()) }
            }
        case .hqDepth:
            avSource.stopWriting()
        }

        // Snapshot the live LiDAR coverage mesh for the post-record inspector (ARKit only).
        // PAUSE the AR session FIRST: scene reconstruction keeps rewriting each ARMeshAnchor's
        // Metal vertex/face buffers on ARKit's queue, and reading them here (MainActor) while they
        // mutate is a data race / EXC_BAD_ACCESS. reset() re-runs the config to resume for a new capture.
        if backend == .arkit {
            session?.pause()
            meshNode = MeshSnapshot.node(from: currentMeshAnchors())
        } else {
            meshNode = nil
        }

        let meta = CaptureMetadata(
            sessionID: sessionID,
            description: captureDescription,
            backend: backend,
            orientation: orientation,
            capturedAt: recordStartedAt,
            finalizedAt: Date(),
            frameCount: 0,                       // overwritten by writer with the authoritative count
            providesPose: backend.providesPose,
            depthSource: backend == .arkit
                ? "ARKit \(coordinator.depthMode) (LiDAR, temporally processed)"
                : "AVFoundation builtInLiDARDepthCamera (raw, absolute, unfiltered)",
            deviceModel: UIDevice.current.model,
            systemVersion: UIDevice.current.systemVersion,
            appVersion: (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "?")
        let confidenceNote = backend == .arkit
            ? "depth valid (255) where ARConfidenceLevel >= medium; else NaN / 0."
            : "no per-pixel confidence; depth valid where finite (>0); holes = NaN / 0."

        let result = await w.finalize(metadata: meta,
                                      trackingWasNormalThroughout: trackingNormal,
                                      confidenceNote: confidenceNote)
        writer = nil

        if result.frameCount == 0 {
            phase = .failed("no frames captured — move slower and keep the subject in view")
        } else {
            if !result.errors.isEmpty {
                trackingMessage = "\(result.errors.count) write warning(s); first: \(result.errors[0])"
            }
            exportURL = nil
            uploadMessage = ""
            phase = .finished(result.captureDir)
            let dir = result.captureDir
            Task {
                let zip = await Task.detached { Exporter.zip(directory: dir) }.value
                self.exportURL = zip
            }
        }
    }

    // MARK: - transmit (review screen; nothing is auto-sent)

    /// Upload the finished session's zip to the Linux receiver. No-op until the zip is ready.
    func transmit() {
        guard let zip = exportURL else { uploadMessage = "still zipping…"; return }
        guard UploadConfig.isConfigured else {
            uploadMessage = "set server URL + token in Settings first"; return
        }
        switch Uploader.shared.upload(zipURL: zip, sessionID: sessionID) {
        case .success:  uploadMessage = "uploading in background…"
        case .failure(let e): uploadMessage = "upload failed: \(e.localizedDescription)"
        }
    }

    // MARK: - source callbacks (MainActor)

    func note(frameCount: Int, elapsed: TimeInterval, validFraction: Double) {
        self.frameCount = frameCount
        self.elapsed = elapsed
        self.validDepthFraction = validFraction
    }

    func note(tracking: String) { trackingMessage = tracking }

    func fail(_ message: String) {
        if case .finished = phase { return }
        phase = .failed(message)
    }

    /// Return to the live preview for another capture ("discard & re-record" also lands here).
    func reset() {
        exportURL = nil
        uploadMessage = ""
        frameCount = 0; elapsed = 0; validDepthFraction = 0
        meshNode = nil
        switch backend {
        case .arkit:
            // resume the AR session (it was paused at stop for the safe mesh snapshot); run()
            // with resetTracking gives a fresh world/mesh for the next capture.
            if session != nil { runConfiguration() }
            phase = session == nil ? .idle : .previewing
        case .hqDepth:
            avSource.startPreview()          // re-arm the HQ session (recovers from a transient failure)
            phase = .previewing
        }
    }

    /// The actual interface orientation at capture time, recorded into metadata. The UI now
    /// rotates freely (Info.plist allows portrait + landscape); saved buffers/intrinsics stay
    /// sensor-native regardless of it (IOS_NOTES §6), so this is descriptive metadata only.
    private func currentInterfaceOrientation() -> CaptureOrientation {
        let scene = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first
        return (scene?.interfaceOrientation.isPortrait ?? false) ? .portrait : .landscape
    }

    // MARK: - paths

    private static func makeSessionID() -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyyMMdd_HHmmss"
        return "session_" + f.string(from: Date())
    }

    /// `Documents/sessions/<id>/capture/` — visible in the Files app via the Info.plist keys.
    static func captureDirectory(for id: String) throws -> URL {
        let docs = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                               appropriateFor: nil, create: true)
        return docs.appendingPathComponent("sessions", isDirectory: true)
            .appendingPathComponent(id, isDirectory: true)
            .appendingPathComponent("capture", isDirectory: true)
    }
}

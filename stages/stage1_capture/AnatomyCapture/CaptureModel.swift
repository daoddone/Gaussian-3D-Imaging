import Foundation
import ARKit
import CoreImage
import CoreGraphics
import Metal
import Observation

/// Owns the capture lifecycle and the UI state. `@MainActor` (so SwiftUI reads
/// are safe) and `@Observable`. The `ARSession` itself is created and owned by
/// the RealityKit `ARView` preview and handed here via `bind(session:)`; this
/// model configures/runs it, drives the `SessionCoordinator`, and finalizes the
/// `FrameWriter`.
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

    let coordinator: SessionCoordinator

    private var session: ARSession?
    private let delegateQueue = DispatchQueue(label: "ar.delegate", qos: .userInitiated)
    private var writer: FrameWriter?
    private var sessionID: String = ""
    private var safetyStop: Task<Void, Never>?

    let budgetSeconds: TimeInterval = 20

    init() {
        let context: CIContext = {
            if let device = MTLCreateSystemDefaultDevice() {
                return CIContext(mtlDevice: device)
            }
            return CIContext()
        }()
        let srgb = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        coordinator = SessionCoordinator(ciContext: context, colorSpace: srgb,
                                         confidenceThreshold: 1, depthMode: "smoothedSceneDepth")
        coordinator.model = self
    }

    /// LiDAR + depth availability on this device.
    static func isSupported() -> Bool {
        ARWorldTrackingConfiguration.isSupported &&
        (ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) ||
         ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth))
    }

    /// Called by the ARView preview once the session exists: wire the delegate
    /// and start the live camera/tracking (preview is live before recording).
    func bind(session: ARSession) {
        guard self.session == nil else { return }
        self.session = session
        session.delegate = coordinator
        session.delegateQueue = delegateQueue
        runConfiguration()
        if case .idle = phase { phase = .previewing }
    }

    private func runConfiguration() {
        let cfg = ARWorldTrackingConfiguration()
        cfg.worldAlignment = .gravity                 // Y up; world yaw arbitrary at start
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            cfg.frameSemantics = [.smoothedSceneDepth]
            coordinator.depthMode = "smoothedSceneDepth"
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            cfg.frameSemantics = [.sceneDepth]
            coordinator.depthMode = "sceneDepth"
        }
        session?.run(cfg, options: [.resetTracking, .removeExistingAnchors])
    }

    // MARK: - recording

    func startRecording() {
        guard phase == .previewing || phase == .idle else { return }
        do {
            sessionID = Self.makeSessionID()
            let dir = try Self.captureDirectory(for: sessionID)
            let w = try FrameWriter(captureDir: dir)
            writer = w
            frameCount = 0; elapsed = 0; validDepthFraction = 0
            phase = .recording
            delegateQueue.async { [coordinator] in coordinator.startWriting(writer: w) }
            // safety net; the coordinator also self-stops at the budget/cap.
            safetyStop = Task { [weak self] in
                try? await Task.sleep(nanoseconds: UInt64((self?.budgetSeconds ?? 20) + 1) * 1_000_000_000)
                if self?.phase == .recording { await self?.stopRecording() }
            }
        } catch {
            phase = .failed("could not start capture: \(error.localizedDescription)")
        }
    }

    /// Called by the coordinator (via MainActor hop) when the budget/cap is hit.
    func finishFromBudget() {
        Task { await stopRecording() }
    }

    func stopRecording() async {
        guard phase == .recording else { return }
        safetyStop?.cancel()
        phase = .finalizing
        guard let w = writer else { phase = .failed("no writer"); return }

        // Stop appends on the delegate queue (serializes after any in-flight
        // frame), then finalize on the writer's io queue (runs after all appends).
        let trackingNormal: Bool = await withCheckedContinuation { cont in
            delegateQueue.async { [coordinator] in cont.resume(returning: coordinator.stopWriting()) }
        }
        let result = await w.finalize(sessionID: sessionID, depthMode: coordinator.depthMode,
                                      confidenceThreshold: "medium",
                                      trackingWasNormalThroughout: trackingNormal)
        writer = nil

        if result.frameCount == 0 {
            phase = .failed("no frames captured — move slower and keep the subject in view")
        } else {
            if !result.errors.isEmpty {
                trackingMessage = "\(result.errors.count) write warning(s); first: \(result.errors[0])"
            }
            exportURL = nil
            phase = .finished(result.captureDir)
            // Build a shareable .zip off the main actor; the folder is already in
            // Files regardless. exportURL flips on when the zip is ready.
            let dir = result.captureDir
            Task {
                let zip = await Task.detached { Exporter.zip(directory: dir) }.value
                self.exportURL = zip
            }
        }
    }

    // MARK: - coordinator callbacks (MainActor)

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

    /// Return to the live preview for another capture.
    func reset() {
        exportURL = nil
        frameCount = 0; elapsed = 0; validDepthFraction = 0
        phase = session == nil ? .idle : .previewing
    }

    // MARK: - paths

    private static func makeSessionID() -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyyMMdd_HHmmss"
        return "session_" + f.string(from: Date())
    }

    /// `Documents/sessions/<id>/capture/` — visible in the Files app via the
    /// Info.plist file-sharing keys, ready to copy into the pipeline's sessions/.
    static func captureDirectory(for id: String) throws -> URL {
        let docs = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                               appropriateFor: nil, create: true)
        return docs.appendingPathComponent("sessions", isDirectory: true)
            .appendingPathComponent(id, isDirectory: true)
            .appendingPathComponent("capture", isDirectory: true)
    }
}

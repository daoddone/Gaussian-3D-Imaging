import SwiftUI
import ARKit
import RealityKit

/// ARKit live preview (RealityKit `ARView` owns the `ARSession`). `dismantleUIView` pauses the
/// session when SwiftUI swaps to the AVFoundation preview, so only one backend owns the camera.
struct ARViewContainer: UIViewRepresentable {
    let model: CaptureModel
    let showOverlay: Bool          // draw the LiDAR coverage mesh, or plain camera video

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        view.environment.background = .cameraFeed()
        applyOverlay(view)
        model.bind(session: view.session)
        return view
    }

    // Re-invoked whenever showOverlay changes (SwiftUI observes model.showOverlay via the parent).
    func updateUIView(_ uiView: ARView, context: Context) { applyOverlay(uiView) }

    /// Overlay is display-only: toggling debugOptions never touches the saved capture buffers.
    private func applyOverlay(_ v: ARView) {
        if showOverlay { v.debugOptions.insert(.showSceneUnderstanding) }
        else { v.debugOptions.remove(.showSceneUnderstanding) }
    }

    static func dismantleUIView(_ uiView: ARView, coordinator: ()) {
        uiView.session.pause()
    }
}

/// Centered reticle + crosshair guidance overlay.
struct ReticleOverlay: View {
    var body: some View {
        GeometryReader { geo in
            let d = min(geo.size.width, geo.size.height) * 0.42
            ZStack {
                Circle().stroke(.white.opacity(0.85), lineWidth: 2).frame(width: d, height: d)
                Path { p in
                    let c = CGPoint(x: geo.size.width / 2, y: geo.size.height / 2)
                    p.move(to: CGPoint(x: c.x - 12, y: c.y)); p.addLine(to: CGPoint(x: c.x + 12, y: c.y))
                    p.move(to: CGPoint(x: c.x, y: c.y - 12)); p.addLine(to: CGPoint(x: c.x, y: c.y + 12))
                }.stroke(.white.opacity(0.85), lineWidth: 2)
            }
            .position(x: geo.size.width / 2, y: geo.size.height / 2)
            .allowsHitTesting(false)
        }
    }
}

struct CaptureView: View {
    @State var model: CaptureModel
    @State private var showSettings = false

    var body: some View {
        ZStack {
            if CaptureModel.isSupported() {
                // Preview branches on the selected backend; swapping tears down the other session.
                if model.backend == .arkit {
                    ARViewContainer(model: model, showOverlay: model.showOverlay).ignoresSafeArea()
                } else {
                    AVCapturePreview(source: model.avSource).ignoresSafeArea()
                }
                ReticleOverlay().ignoresSafeArea()
                overlay
            } else {
                unsupported
            }
        }
        .preferredColorScheme(.dark)
        .statusBarHidden(true)
        .sheet(isPresented: $model.showInspector) {
            CoverageInspectorSheet(meshNode: model.meshNode)
        }
        .sheet(isPresented: $showSettings) { SettingsView() }
    }

    // MARK: - overlay

    private var overlay: some View {
        VStack(spacing: 0) {
            topBar
            Spacer()
            bottomControls
        }
        .padding()
    }

    private var topBar: some View {
        HStack(alignment: .top) {
            Button { showSettings = true } label: {
                Image(systemName: "gearshape.fill").font(.title3).foregroundStyle(.white)
                    .padding(8).background(.black.opacity(0.35), in: Circle())
            }
            .accessibilityLabel("Settings")
            .padding(.trailing, 4)
            VStack(alignment: .leading, spacing: 4) {
                Label(model.trackingMessage.isEmpty ? "starting…" : model.trackingMessage,
                      systemImage: "dot.radiowaves.left.and.right")
                    .font(.caption).foregroundStyle(.white)
                // valid-depth readout shown live in preview AND recording (framing aid). % = share
                // of depth pixels the sensor trusts; see the guidance capsule below for how to raise it.
                if model.validDepthFraction > 0 || model.phase == .recording {
                    Text("valid depth: \(Int(model.validDepthFraction * 100))%")
                        .font(.caption2).foregroundStyle(model.validDepthFraction > 0.4 ? .green : .yellow)
                }
            }
            Spacer()
            // Overlay/video toggle (ARKit only): mesh coverage overlay vs plain camera video.
            // Display-only; safe to toggle any time, including mid-recording.
            if model.backend == .arkit {
                Button { model.showOverlay.toggle() } label: {
                    Image(systemName: model.showOverlay ? "cube.transparent.fill" : "cube.transparent")
                        .font(.title3)
                        .foregroundStyle(model.showOverlay ? .green : .white)
                        .padding(8).background(.black.opacity(0.35), in: Circle())
                }
                .accessibilityLabel(model.showOverlay ? "Hide coverage mesh" : "Show coverage mesh")
                .padding(.trailing, 6)
            }
            // Focus toggle (HQ-Depth only; ARKit manages its own focus). Auto = continuous
            // autofocus, Lock = pinned lens. Tap the preview anywhere to focus that point.
            if model.backend == .hqDepth {
                Button { model.focusLocked.toggle() } label: {
                    Image(systemName: model.focusLocked ? "lock.circle.fill" : "a.circle")
                        .font(.title3)
                        .foregroundStyle(model.focusLocked ? .yellow : .white)
                        .padding(8).background(.black.opacity(0.35), in: Circle())
                }
                .accessibilityLabel(model.focusLocked ? "Focus locked — tap to resume autofocus"
                                                      : "Autofocus — tap to lock focus")
                .padding(.trailing, 6)
            }
            if model.phase == .recording {
                VStack(alignment: .trailing, spacing: 4) {
                    Text("\(model.frameCount) frames").font(.headline).foregroundStyle(.white)
                    Text(String(format: "%.1f / %.0f s", model.elapsed, model.budgetSeconds))
                        .font(.caption).foregroundStyle(.white)
                }
            }
        }
        .padding(.horizontal, 8).padding(.vertical, 6)
        .background(.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
    }

    /// Framework + orientation toggles + description, shown before recording (locked during).
    private var setupControls: some View {
        VStack(spacing: 8) {
            Picker("Framework", selection: Binding(
                get: { model.backend },
                set: { model.setBackend($0) })) {
                ForEach(CaptureBackend.allCases, id: \.self) { Text($0.uiLabel).tag($0) }
            }.pickerStyle(.segmented)

            // Framing orientation is auto-detected from how the phone is held at record time
            // (the UI now rotates freely); no manual picker needed. Saved data stays sensor-native.

            TextField("Description (e.g. left forearm flap, post-debridement)",
                      text: $model.captureDescription)
                .textFieldStyle(.roundedBorder).font(.caption)
        }
        .padding(10)
        .background(.black.opacity(0.4), in: RoundedRectangle(cornerRadius: 12))
        .frame(maxWidth: 420)
    }

    @ViewBuilder private var bottomControls: some View {
        switch model.phase {
        case .idle, .previewing:
            VStack(spacing: 10) {
                setupControls
                Text(guidanceText)
                    .font(.caption).foregroundStyle(.white).multilineTextAlignment(.center)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6).background(.black.opacity(0.35), in: Capsule())
                recordButton
            }
        case .recording:
            VStack(spacing: 10) {
                ProgressView(value: min(model.elapsed / model.budgetSeconds, 1))
                    .tint(.red).frame(maxWidth: 360)
                Button(role: .destructive) { Task { await model.stopRecording() } } label: {
                    Label("Stop", systemImage: "stop.fill").font(.title3.bold())
                        .padding(.horizontal, 28).padding(.vertical, 12)
                        .background(.red, in: Capsule()).foregroundStyle(.white)
                }
            }
        case .finalizing:
            ProgressView("Saving…").tint(.white).foregroundStyle(.white)
                .padding().background(.black.opacity(0.4), in: RoundedRectangle(cornerRadius: 12))
        case .finished(let dir):
            reviewControls(dir: dir)
        case .failed(let message):
            VStack(spacing: 10) {
                Text(message).font(.callout).foregroundStyle(.white).multilineTextAlignment(.center)
                    .padding().background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 12))
                Button("Try again") { model.reset() }.buttonStyle(.borderedProminent)
            }
        }
    }

    /// Framing guidance that doubles as the valid-depth explainer (owner question #4). "valid depth"
    /// = share of depth pixels the sensor trusts. ARKit temporally FILLS + confidence-thresholds its
    /// depth, so it reads high at ~1–2 ft; HQ is RAW LiDAR (holes = NaN), so it needs the subject
    /// past the ~25 cm near-field to read high. When it's low, tell the clinician how to raise it.
    private var guidanceText: String {
        let low = model.validDepthFraction > 0 && model.validDepthFraction < 0.4
        if model.backend == .hqDepth {
            return low
                ? "Low valid depth — back off to ~30 cm+. Raw LiDAR reads holes (NaN) closer than ~25 cm."
                : "Center the region, hold ~30 cm, orbit slowly (~20 s). Tap to focus."
        } else {
            return low
                ? "Low valid depth — hold ~1–2 ft and fill the frame; ARKit is most confident there."
                : "Center the region, hold ~30 cm, orbit slowly (finish in ~20 s)."
        }
    }

    private var recordButton: some View {
        Button { model.startRecording() } label: {
            ZStack {
                Circle().strokeBorder(.white, lineWidth: 4).frame(width: 78, height: 78)
                Circle().fill(.red).frame(width: 62, height: 62)
            }
        }
        .accessibilityLabel("Start recording")
    }

    /// Review-before-send: transmit, share, or discard & re-record. Nothing is auto-sent.
    private func reviewControls(dir: URL) -> some View {
        VStack(spacing: 10) {
            Label("\(model.frameCount) frames — \(model.backend.uiLabel)", systemImage: "checkmark.circle.fill")
                .font(.headline).foregroundStyle(.green)
            // Post-record LiDAR coverage cloud — available for BOTH backends now (ARKit: fused
            // world cloud; HQ: single-view). Shown whenever a cloud was captured.
            if model.meshNode != nil {
                Button { model.showInspector = true } label: {
                    Label("Inspect 3D coverage", systemImage: "cube.transparent")
                        .font(.subheadline).padding(.horizontal, 16).padding(.vertical, 8)
                        .background(.white.opacity(0.2), in: Capsule()).foregroundStyle(.white)
                }
            }
            TextField("Description", text: $model.captureDescription)
                .textFieldStyle(.roundedBorder).font(.caption).frame(maxWidth: 420)
                // metadata.json was written at finalize; persist edits made here so the on-disk
                // capture dir (what AirDrop→Files copies) always carries the current description.
                .onChange(of: model.captureDescription) { _, _ in model.syncDescriptionToDisk() }
            if !model.uploadMessage.isEmpty {
                Text(model.uploadMessage).font(.caption2).foregroundStyle(.yellow)
            }
            HStack(spacing: 12) {
                Button { model.transmit() } label: {
                    Label("Transmit", systemImage: "antenna.radiowaves.left.and.right")
                        .padding(.horizontal, 18).padding(.vertical, 10)
                        .background(model.exportURL == nil ? .gray : .blue, in: Capsule())
                        .foregroundStyle(.white)
                }.disabled(model.exportURL == nil)

                if let url = model.exportURL {
                    ShareLink(item: url) {
                        Image(systemName: "square.and.arrow.up")
                            .padding(12).background(.white.opacity(0.2), in: Circle()).foregroundStyle(.white)
                    }
                }
                Button("Discard & re-record") { model.reset() }
                    .padding(.horizontal, 18).padding(.vertical, 10)
                    .background(.white.opacity(0.2), in: Capsule()).foregroundStyle(.white)
            }
            Text("In Files: On My iPhone → AnatomyCapture → sessions → \(dir.deletingLastPathComponent().lastPathComponent)")
                .font(.caption2).foregroundStyle(.white.opacity(0.7)).multilineTextAlignment(.center)
        }
        .padding().background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 14))
    }

    private var unsupported: some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill").font(.largeTitle).foregroundStyle(.yellow)
            Text("This device has no LiDAR scene depth.")
                .font(.headline).foregroundStyle(.white)
            Text("AnatomyCapture needs an iPhone Pro / iPad Pro with a rear LiDAR sensor (e.g. iPhone 14 Pro).")
                .font(.caption).foregroundStyle(.white).multilineTextAlignment(.center).padding()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity).background(.black)
    }
}

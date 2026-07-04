import SwiftUI
import ARKit
import RealityKit

/// ARKit live preview (RealityKit `ARView` owns the `ARSession`). `dismantleUIView` pauses the
/// session when SwiftUI swaps to the AVFoundation preview, so only one backend owns the camera.
struct ARViewContainer: UIViewRepresentable {
    let model: CaptureModel

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        view.environment.background = .cameraFeed()
        model.bind(session: view.session)
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

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

    var body: some View {
        ZStack {
            if CaptureModel.isSupported() {
                // Preview branches on the selected backend; swapping tears down the other session.
                if model.backend == .arkit {
                    ARViewContainer(model: model).ignoresSafeArea()
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
            VStack(alignment: .leading, spacing: 4) {
                Label(model.trackingMessage.isEmpty ? "starting…" : model.trackingMessage,
                      systemImage: "dot.radiowaves.left.and.right")
                    .font(.caption).foregroundStyle(.white)
                if model.phase == .recording {
                    Text("valid depth: \(Int(model.validDepthFraction * 100))%")
                        .font(.caption2).foregroundStyle(model.validDepthFraction > 0.4 ? .green : .yellow)
                }
            }
            Spacer()
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

            Picker("Framing", selection: $model.orientation) {
                ForEach(CaptureOrientation.allCases, id: \.self) { Text($0.uiLabel).tag($0) }
            }.pickerStyle(.segmented)

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
                Text("Center the region, hold ~30 cm, orbit slowly (finish in ~20 s).")
                    .font(.caption).foregroundStyle(.white).padding(.horizontal, 12)
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
            TextField("Description", text: $model.captureDescription)
                .textFieldStyle(.roundedBorder).font(.caption).frame(maxWidth: 420)
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

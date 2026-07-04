import SwiftUI
import SceneKit

/// Full-screen interactive viewer for the post-record coverage mesh. `allowsCameraControl`
/// gives pinch-zoom / rotate / pan for free, so the clinician can orbit the reconstructed
/// surface and spot under-sampled regions (holes) before deciding to re-record.
struct CoverageMeshView: UIViewRepresentable {
    let meshNode: SCNNode

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        let scene = SCNScene()
        view.scene = scene
        view.allowsCameraControl = true          // pinch/zoom/rotate/pan
        view.autoenablesDefaultLighting = true
        view.backgroundColor = .black
        view.antialiasingMode = .multisampling4X

        // Center the mesh at the origin so the default camera framing lands on it.
        let (minB, maxB) = meshNode.boundingBox
        let center = SCNVector3((minB.x + maxB.x) / 2, (minB.y + maxB.y) / 2, (minB.z + maxB.z) / 2)
        meshNode.pivot = SCNMatrix4MakeTranslation(center.x, center.y, center.z)
        scene.rootNode.addChildNode(meshNode)

        // A camera looking at it; allowsCameraControl takes over from here.
        let cam = SCNNode()
        cam.camera = SCNCamera()
        let extent = max(maxB.x - minB.x, maxB.y - minB.y, maxB.z - minB.z)
        cam.position = SCNVector3(0, 0, max(0.3, extent * 2.2))
        scene.rootNode.addChildNode(cam)
        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {}
}

/// Sheet wrapper: the inspector plus guidance + a close button.
struct CoverageInspectorSheet: View {
    let meshNode: SCNNode?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack(alignment: .top) {
            if let node = meshNode {
                CoverageMeshView(meshNode: node).ignoresSafeArea()
            } else {
                VStack(spacing: 10) {
                    Image(systemName: "cube.transparent").font(.largeTitle).foregroundStyle(.gray)
                    Text("No coverage mesh").font(.headline).foregroundStyle(.white)
                    Text("The 3D inspector uses ARKit's live LiDAR mesh — available in ARKit mode only, and only if the scan ran long enough to build one.")
                        .font(.caption).foregroundStyle(.white.opacity(0.7))
                        .multilineTextAlignment(.center).padding(.horizontal, 24)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity).background(.black)
            }
            HStack {
                Text("Orbit to check coverage — dark gaps are under-sampled.")
                    .font(.caption).foregroundStyle(.white)
                    .padding(.horizontal, 10).padding(.vertical, 6)
                    .background(.black.opacity(0.4), in: Capsule())
                Spacer()
                Button("Done") { dismiss() }.buttonStyle(.borderedProminent)
            }
            .padding()
        }
        .preferredColorScheme(.dark)
    }
}

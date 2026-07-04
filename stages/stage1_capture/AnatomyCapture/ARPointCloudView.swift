import SwiftUI
import ARKit
import SceneKit

/// ARKit live preview via `ARSCNView` (SceneKit) with a LIVE accumulated LiDAR point-cloud overlay.
///
/// `SessionCoordinator` fuses per-frame depth into WORLD points (it has the metric pose); this view
/// refreshes a SceneKit point node ~2.5×/s from that growing cloud, so the clinician watches coverage
/// fill in and can dwell on sparse areas — the during-recording feedback that matters most.
///
/// Replaces the RealityKit `ARView` + scene-mesh wireframe (which rendered occluded geometry through
/// foreground objects, and which RealityKit can't draw as points). The capture DATA path is unchanged:
/// `model.bind(session:)` still makes `SessionCoordinator` the `ARSessionDelegate`. ARSCNView renders
/// the camera feed from `session.currentFrame`, independent of who the session delegate is; our
/// `ARSCNViewDelegate.renderer(_:updateAtTime:)` is a separate hook we use only to refresh the overlay.
struct ARPointCloudView: UIViewRepresentable {
    let model: CaptureModel
    let showCloud: Bool

    /// Confined to the main / SceneKit-render thread (SwiftUI make/update/dismantle + the render
    /// callback all run there); `@unchecked Sendable` records that so the delegate can be retained.
    final class Coordinator: NSObject, ARSCNViewDelegate, @unchecked Sendable {
        let cloudNode = SCNNode()
        var showCloud = true
        weak var source: SessionCoordinator?
        private var lastBuild: TimeInterval = 0

        func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
            guard time - lastBuild > 0.4 else { return }        // throttle to ~2.5 Hz
            lastBuild = time
            guard showCloud else { if cloudNode.geometry != nil { cloudNode.geometry = nil }; return }
            guard let pts = source?.displayCloudSnapshot(), pts.count >= 16 else { return }
            cloudNode.geometry = PointCloudNode.geometry(points: pts)   // world-space; aligns via the AR camera
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> ARSCNView {
        let v = ARSCNView(frame: .zero)
        let scene = SCNScene()                                 // explicit non-nil scene (ARSCNView.scene is optional)
        v.scene = scene
        v.automaticallyUpdatesLighting = true
        v.delegate = context.coordinator                       // render-loop hook (overlay refresh)
        scene.rootNode.addChildNode(context.coordinator.cloudNode)
        context.coordinator.source = model.coordinator
        context.coordinator.showCloud = showCloud
        model.bind(session: v.session)                         // SessionCoordinator becomes the ARSessionDelegate
        return v
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {
        context.coordinator.showCloud = showCloud
        if !showCloud { context.coordinator.cloudNode.geometry = nil }
    }

    static func dismantleUIView(_ uiView: ARSCNView, coordinator: Coordinator) {
        uiView.session.pause()
        uiView.delegate = nil
    }
}

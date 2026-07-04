import SceneKit
import simd

/// Builds a SceneKit point-cloud node from accumulated 3D points, for the post-record coverage
/// inspector. A LiDAR point cloud is far denser and truer to a close-range subject than ARKit's
/// fused room-scale mesh (owner reports #1/#3), and — unlike the mesh — is available for the
/// HQ-Depth path too (single-view, no pose). Feedback only; never part of the saved capture.
enum PointCloudNode {
    /// `points` are world-space (ARKit) or camera-space (HQ, single view). Colored by height so the
    /// shape reads in the orbit viewer. Returns nil if there are too few points to show.
    static func make(points: [SIMD3<Float>]) -> SCNNode? {
        guard points.count >= 16 else { return nil }

        let vertexData = points.withUnsafeBytes { Data($0) }
        let vertexSource = SCNGeometrySource(
            data: vertexData, semantic: .vertex, vectorCount: points.count,
            usesFloatComponents: true, componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
            dataStride: MemoryLayout<SIMD3<Float>>.stride)      // 16-byte stride skips SIMD3 padding

        // Per-point color ramp by height (y) for legibility.
        let ys = points.map(\.y)
        let ymin = ys.min() ?? 0
        let span = max((ys.max() ?? 1) - ymin, 1e-4)
        var colors = [SIMD4<Float>]()
        colors.reserveCapacity(points.count)
        for p in points {
            let t = (p.y - ymin) / span
            colors.append(SIMD4<Float>(0.25 + 0.75 * t, 0.65, 1.0 - 0.55 * t, 1))
        }
        let colorData = colors.withUnsafeBytes { Data($0) }
        let colorSource = SCNGeometrySource(
            data: colorData, semantic: .color, vectorCount: colors.count,
            usesFloatComponents: true, componentsPerVector: 4,
            bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
            dataStride: MemoryLayout<SIMD4<Float>>.stride)

        let indices = (0..<UInt32(points.count)).map { $0 }
        let indexData = indices.withUnsafeBytes { Data($0) }
        let element = SCNGeometryElement(
            data: indexData, primitiveType: .point, primitiveCount: points.count,
            bytesPerIndex: MemoryLayout<UInt32>.size)
        element.pointSize = 6
        element.minimumPointScreenSpaceRadius = 1.5
        element.maximumPointScreenSpaceRadius = 6

        let geo = SCNGeometry(sources: [vertexSource, colorSource], elements: [element])
        // A freshly built SCNGeometry has NO material, so `firstMaterial` is nil and setting through
        // it is a no-op — attach one explicitly (as MeshSnapshot does). `.constant` = unlit, so the
        // per-vertex height colors show as-is instead of rendering dark under default lighting.
        let mat = SCNMaterial()
        mat.lightingModel = .constant
        mat.isDoubleSided = true
        geo.materials = [mat]
        return SCNNode(geometry: geo)
    }
}

import SceneKit
import simd

/// Builds SceneKit point-cloud geometry from 3D points.
///  • `geometry(points:)` — raw world-space geometry for the LIVE overlay (points stay put so the
///    cloud is stable in the AR scene as it accumulates).
///  • `make(points:)` — for the POST-record inspector: robustly recenters on the dense region and
///    crops far outliers first, so the orbit view frames the subject (owner report: was room-scale)
///    instead of the bounding-box center. Feedback only; never part of the saved capture.
enum PointCloudNode {

    /// Median-center + crop the farthest ~8% (outliers/stragglers), translated to the origin so the
    /// inspector's bounding-box framing lands on the subject. Returns points as-is if too few.
    static func centerAndCrop(_ points: [SIMD3<Float>]) -> [SIMD3<Float>] {
        guard points.count > 32 else { return points }
        func median(_ v: [Float]) -> Float { let s = v.sorted(); return s[s.count / 2] }
        let center = SIMD3<Float>(median(points.map(\.x)), median(points.map(\.y)), median(points.map(\.z)))
        let dists = points.map { simd_distance($0, center) }.sorted()
        let thresh = dists[min(dists.count - 1, Int(Float(dists.count) * 0.92))]
        return points.compactMap { simd_distance($0, center) <= thresh ? $0 - center : nil }
    }

    /// Live-overlay node (world-space, no recentering) — swap its `.geometry` as the cloud grows.
    static func liveNode() -> SCNNode { SCNNode() }

    /// Post-record inspector node: recentered + cropped for good framing.
    static func make(points: [SIMD3<Float>]) -> SCNNode? {
        guard let geo = geometry(points: centerAndCrop(points)) else { return nil }
        return SCNNode(geometry: geo)
    }

    /// Core builder: a `.point` SCNGeometry with a per-vertex height color ramp. Safe to call off the
    /// main thread (pure geometry construction). Returns nil for too few points.
    static func geometry(points: [SIMD3<Float>]) -> SCNGeometry? {
        guard points.count >= 16 else { return nil }

        let vertexData = points.withUnsafeBytes { Data($0) }
        let vertexSource = SCNGeometrySource(
            data: vertexData, semantic: .vertex, vectorCount: points.count,
            usesFloatComponents: true, componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
            dataStride: MemoryLayout<SIMD3<Float>>.stride)     // 16-byte stride skips SIMD3 padding

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
        let mat = SCNMaterial()
        mat.lightingModel = .constant          // points carry their own color; render unlit
        mat.isDoubleSided = true
        geo.materials = [mat]
        return geo
    }
}

import Foundation
import ARKit
import SceneKit
import simd

/// Converts the ARKit LiDAR scene-reconstruction anchors captured during a recording into a
/// SceneKit node the clinician can pinch/zoom/rotate afterwards to judge coverage — gaps/holes
/// in the surface reveal under-sampled regions (wound bed / medial arm) before deciding to
/// re-record. Feedback only; never written to the capture. ARKit-only (HQ-Depth has no meshing).
enum MeshSnapshot {

    /// Build a parent `SCNNode` with one child per mesh anchor (positioned by its world transform).
    /// Returns nil if there is no geometry (e.g. HQ-Depth backend, or scan too short to mesh).
    static func node(from anchors: [ARMeshAnchor]) -> SCNNode? {
        let parent = SCNNode()
        var any = false
        for anchor in anchors {
            guard let g = geometry(from: anchor.geometry) else { continue }
            let child = SCNNode(geometry: g)
            child.simdTransform = anchor.transform            // place in world (don't move verts)
            parent.addChildNode(child)
            any = true
        }
        return any ? parent : nil
    }

    private static func geometry(from mesh: ARMeshGeometry) -> SCNGeometry? {
        let vsrc = mesh.vertices
        guard vsrc.count > 0 else { return nil }
        // Vertices: interleaved float3 in a Metal buffer (respect offset/stride).
        var verts = [SCNVector3](); verts.reserveCapacity(vsrc.count)
        let vbase = vsrc.buffer.contents()
        for i in 0..<vsrc.count {
            let p = vbase.advanced(by: vsrc.offset + i * vsrc.stride)
                .assumingMemoryBound(to: (Float, Float, Float).self).pointee
            verts.append(SCNVector3(p.0, p.1, p.2))
        }
        // Faces: triangle index list (ARKit uses 4-byte indices, 3 per primitive).
        let faces = mesh.faces
        let ibase = faces.buffer.contents()
        let idxCount = faces.count * faces.indexCountPerPrimitive
        var indices = [Int32](repeating: 0, count: idxCount)
        for i in 0..<idxCount {
            indices[i] = ibase.advanced(by: i * faces.bytesPerIndex)
                .assumingMemoryBound(to: Int32.self).pointee
        }

        let source = SCNGeometrySource(vertices: verts)
        let element = SCNGeometryElement(indices: indices, primitiveType: .triangles)
        let geo = SCNGeometry(sources: [source], elements: [element])
        let mat = SCNMaterial()
        mat.diffuse.contents = UIColor(white: 0.72, alpha: 1.0)
        mat.isDoubleSided = true                              // see thin/edge regions from both sides
        mat.lightingModel = .physicallyBased
        geo.materials = [mat]
        return geo
    }
}

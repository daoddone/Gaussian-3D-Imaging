import simd

/// Accumulates a coverage point cloud by back-projecting LiDAR depth frames, for the post-record
/// inspector. Uses the SAME convention the capture writes for the pipeline — OpenCV camera space
/// (x right, y down, z forward), world = R·cam + t with R,t = openCVCameraToWorld — so the cloud
/// is oriented like the reconstruction. Voxel-deduped + capped so it stays bounded on-device.
///
/// ARKit: pass the per-frame R,t → a fused world cloud. HQ-Depth (no pose): reset() each frame and
/// pass identity R,t → a single-view (2.5D) cloud of the most recent frame. Feedback only.
final class PointCloudAccumulator {
    private(set) var points: [SIMD3<Float>] = []
    private var voxels = Set<Int64>()
    private let cap: Int
    private let voxel: Float

    init(cap: Int = 200_000, voxelMeters: Float = 0.003) { self.cap = cap; self.voxel = voxelMeters }

    func reset() { points.removeAll(keepingCapacity: true); voxels.removeAll(keepingCapacity: true) }

    /// depth/mask are row-major [H,W] at (dw,dh); K is at color resolution (colorW,colorH) and is
    /// scaled to depth resolution here; R,t are OpenCV camera→world. Subsamples every `step` pixels.
    func add(depth: [Float], mask: [UInt8], dw: Int, dh: Int,
             K: [[Double]], colorW: Int, colorH: Int,
             R: [[Double]], t: [Double], step: Int = 6) {
        guard points.count < cap, dw > 0, dh > 0, colorW > 0, colorH > 0,
              depth.count == dw * dh, mask.count == dw * dh else { return }
        let sx = Double(dw) / Double(colorW), sy = Double(dh) / Double(colorH)
        let fx = K[0][0] * sx, fy = K[1][1] * sy, cx = K[0][2] * sx, cy = K[1][2] * sy
        guard fx != 0, fy != 0 else { return }
        let r00 = Float(R[0][0]), r01 = Float(R[0][1]), r02 = Float(R[0][2])
        let r10 = Float(R[1][0]), r11 = Float(R[1][1]), r12 = Float(R[1][2])
        let r20 = Float(R[2][0]), r21 = Float(R[2][1]), r22 = Float(R[2][2])
        let tx = Float(t[0]), ty = Float(t[1]), tz = Float(t[2])
        let inv = 1.0 / voxel
        var v = 0
        while v < dh {
            var u = 0
            while u < dw {
                let idx = v * dw + u
                if mask[idx] == 255 {
                    let d = Double(depth[idx])
                    let xc = Float((Double(u) - cx) / fx * d)
                    let yc = Float((Double(v) - cy) / fy * d)
                    let zc = Float(d)
                    let wx = r00 * xc + r01 * yc + r02 * zc + tx
                    let wy = r10 * xc + r11 * yc + r12 * zc + ty
                    let wz = r20 * xc + r21 * yc + r22 * zc + tz
                    let xi = Int64((wx * inv).rounded())
                    let yi = Int64((wy * inv).rounded())
                    let zi = Int64((wz * inv).rounded())
                    let key = (xi &* 73_856_093) ^ (yi &* 19_349_663) ^ (zi &* 83_492_791)
                    if voxels.insert(key).inserted {
                        points.append(SIMD3<Float>(wx, wy, wz))
                        if points.count >= cap { return }
                    }
                }
                u += step
            }
            v += step
        }
    }

    /// Identity extrinsics for the HQ single-view cloud (no pose).
    static let identityR: [[Double]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    static let zeroT: [Double] = [0, 0, 0]
}

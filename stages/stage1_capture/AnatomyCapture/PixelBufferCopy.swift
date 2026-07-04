import Foundation
import CoreVideo
import CoreImage
import CoreGraphics
import simd

/// A fully value-copied frame ready to hand to the writer. It carries an
/// immutable `CGImage` (thread-safe) plus plain value arrays, so it is safe to
/// pass across the delegate queue → writer queue boundary. Marked
/// `@unchecked Sendable` because its only reference member, `CGImage`, is an
/// immutable Core Graphics object.
struct FramePayload: @unchecked Sendable {
    let index: Int
    let color: CGImage          // already YCbCr→RGB, sensor-native orientation
    let colorW: Int
    let colorH: Int
    let depth: [Float]          // meters, NaN where invalid, C-order [H,W]
    let mask: [UInt8]           // 255 valid / 0 invalid, [H,W]
    let depthW: Int
    let depthH: Int
    let R: [[Double]]?          // OpenCV camera_to_world rotation (row-major); nil for AVFoundation (no pose)
    let t: [Double]?            // camera position, meters; nil for AVFoundation
    let K: [[Double]]           // intrinsics at color resolution
    let time: Double            // seconds, relative to first kept frame
    let validDepthFraction: Double
}

/// Deep copies out of ARKit `CVPixelBuffer`s. Must be called on the ARSession
/// delegate queue; it copies everything it needs and never retains the buffers
/// or the `ARFrame` (retaining frames stalls ARKit's reuse pool).
enum PixelBufferCopy {

    /// Convert the biplanar YCbCr color buffer to an RGB `CGImage` in the
    /// buffer's native (un-rotated) orientation. `createCGImage` yields a
    /// standard top-left-origin image, matching the depth buffer's layout.
    static func colorCGImage(_ pb: CVPixelBuffer, ctx: CIContext, colorSpace: CGColorSpace) -> CGImage? {
        let ci = CIImage(cvPixelBuffer: pb)
        return ctx.createCGImage(ci, from: ci.extent, format: .RGBA8, colorSpace: colorSpace)
    }

    /// Copy the depth map into `[Float]` (meters) and synthesize the validity
    /// mask + NaN-invalid depth from the confidence map. ARKit depth is dense
    /// (never NaN); validity is signalled only by `confidenceMap`.
    ///
    /// valid = depth.isFinite && depth > 0 && confidence >= threshold
    /// (threshold 1 == `.medium`). Invalid → depth NaN, mask 0.
    static func depthAndMask(depth: CVPixelBuffer,
                             confidence: CVPixelBuffer?,
                             threshold: UInt8) -> (depth: [Float], mask: [UInt8], w: Int, h: Int) {
        let w = CVPixelBufferGetWidth(depth)
        let h = CVPixelBufferGetHeight(depth)

        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        let dRow = CVPixelBufferGetBytesPerRow(depth)
        guard let dBase = CVPixelBufferGetBaseAddress(depth) else {
            return ([Float](repeating: .nan, count: w * h), [UInt8](repeating: 0, count: w * h), w, h)
        }

        // Only trust the confidence buffer if it matches the depth dimensions.
        var conf: CVPixelBuffer? = confidence
        if let c = confidence,
           CVPixelBufferGetWidth(c) != w || CVPixelBufferGetHeight(c) != h {
            conf = nil
        }
        var cRow = 0
        var cBase: UnsafeMutableRawPointer?
        if let c = conf {
            CVPixelBufferLockBaseAddress(c, .readOnly)
            cRow = CVPixelBufferGetBytesPerRow(c)
            cBase = CVPixelBufferGetBaseAddress(c)
        }
        defer { if let c = conf { CVPixelBufferUnlockBaseAddress(c, .readOnly) } }

        var out = [Float](repeating: .nan, count: w * h)
        var mask = [UInt8](repeating: 0, count: w * h)
        for y in 0..<h {
            let drow = dBase.advanced(by: y * dRow).assumingMemoryBound(to: Float32.self)
            let crow = cBase?.advanced(by: y * cRow).assumingMemoryBound(to: UInt8.self)
            let base = y * w
            for x in 0..<w {
                let d = drow[x]
                let conf = crow?[x] ?? 2   // no confidence buffer → treat as high
                if d.isFinite && d > 0 && conf >= threshold {
                    out[base + x] = d
                    mask[base + x] = 255
                }
            }
        }
        return (out, mask, w, h)
    }

    /// Cheap valid-depth fraction for the LIVE PREVIEW readout — subsampled every `step` pixels,
    /// no array allocation. ARKit variant: valid = finite depth > 0 AND confidence >= threshold
    /// (same rule as `depthAndMask`, so the preview % matches what recording would keep).
    static func arkitValidFraction(depth: CVPixelBuffer, confidence: CVPixelBuffer?,
                                   threshold: UInt8, step: Int = 4) -> Double {
        let w = CVPixelBufferGetWidth(depth), h = CVPixelBufferGetHeight(depth)
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        let dRow = CVPixelBufferGetBytesPerRow(depth)
        guard let dBase = CVPixelBufferGetBaseAddress(depth) else { return 0 }
        var conf: CVPixelBuffer? = confidence
        if let c = confidence, CVPixelBufferGetWidth(c) != w || CVPixelBufferGetHeight(c) != h { conf = nil }
        var cRow = 0
        var cBase: UnsafeMutableRawPointer?
        if let c = conf {
            CVPixelBufferLockBaseAddress(c, .readOnly); cRow = CVPixelBufferGetBytesPerRow(c); cBase = CVPixelBufferGetBaseAddress(c)
        }
        defer { if let c = conf { CVPixelBufferUnlockBaseAddress(c, .readOnly) } }
        var valid = 0, total = 0, y = 0
        while y < h {
            let drow = dBase.advanced(by: y * dRow).assumingMemoryBound(to: Float32.self)
            let crow = cBase?.advanced(by: y * cRow).assumingMemoryBound(to: UInt8.self)
            var x = 0
            while x < w {
                let d = drow[x]; let cc = crow?[x] ?? 2
                if d.isFinite && d > 0 && cc >= threshold { valid += 1 }
                total += 1; x += step
            }
            y += step
        }
        return total > 0 ? Double(valid) / Double(total) : 0
    }

    /// Cheap finite-depth fraction for the HQ live-preview readout (subsampled). `depth` must be
    /// DepthFloat32 (convert via AVDepthData.converting(toDepthDataType:) first).
    static func finiteFraction(depthFloat32 depth: CVPixelBuffer, step: Int = 4) -> Double {
        let w = CVPixelBufferGetWidth(depth), h = CVPixelBufferGetHeight(depth)
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        let dRow = CVPixelBufferGetBytesPerRow(depth)
        guard let dBase = CVPixelBufferGetBaseAddress(depth) else { return 0 }
        var valid = 0, total = 0, y = 0
        while y < h {
            let drow = dBase.advanced(by: y * dRow).assumingMemoryBound(to: Float32.self)
            var x = 0
            while x < w { let d = drow[x]; if d.isFinite && d > 0 { valid += 1 }; total += 1; x += step }
            y += step
        }
        return total > 0 ? Double(valid) / Double(total) : 0
    }

    /// AVFoundation depth path: copy a DepthFloat32 map into `[Float]` (meters) and
    /// synthesize the validity mask from finiteness alone. Unlike ARKit, AVCaptureDepthDataOutput
    /// gives no per-pixel confidence map — with filtering disabled, invalid pixels arrive as NaN,
    /// so `valid = depth.isFinite && depth > 0`. The caller MUST pass a buffer already converted
    /// to kCVPixelFormatType_DepthFloat32 (AVDepthData.converting(toDepthDataType:)).
    static func depthFloat32AndFiniteMask(_ depth: CVPixelBuffer) -> (depth: [Float], mask: [UInt8], w: Int, h: Int) {
        let w = CVPixelBufferGetWidth(depth)
        let h = CVPixelBufferGetHeight(depth)
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        let dRow = CVPixelBufferGetBytesPerRow(depth)
        guard let dBase = CVPixelBufferGetBaseAddress(depth) else {
            return ([Float](repeating: .nan, count: w * h), [UInt8](repeating: 0, count: w * h), w, h)
        }
        var out = [Float](repeating: .nan, count: w * h)
        var mask = [UInt8](repeating: 0, count: w * h)
        for y in 0..<h {
            let drow = dBase.advanced(by: y * dRow).assumingMemoryBound(to: Float32.self)
            let base = y * w
            for x in 0..<w {
                let d = drow[x]
                if d.isFinite && d > 0 {
                    out[base + x] = d
                    mask[base + x] = 255
                }
            }
        }
        return (out, mask, w, h)
    }
}

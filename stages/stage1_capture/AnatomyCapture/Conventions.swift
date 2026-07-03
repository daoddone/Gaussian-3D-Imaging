import Foundation
import simd

/// Coordinate-convention conversion for the capture app.
///
/// The whole pipeline uses the OpenCV convention: the camera looks down its +z
/// axis, x right, y down. ARKit's camera frame is OpenGL-style: it looks down
/// -z with +y up (+x right). Converting a camera pose therefore negates the
/// camera's own y and z axes — equivalent to `R_opencv = R_arkit · diag(1,-1,-1)`
/// — while the camera position is unchanged. This mirrors
/// `common/conventions.opencv_c2w_from_arkit` in the Python side and must be
/// confirmed with the orientation self-test.
///
/// `ARCamera.transform` is a camera-to-world matrix in `simd` **column-major**
/// storage (columns 0–2 are the camera basis vectors expressed in world space,
/// column 3 is the translation in meters). `simd` matrix multiplication is
/// ordinary math multiplication, so `transform * diag(1,-1,-1,1)` negates the
/// second and third basis columns, which is exactly the axis flip we want.
enum Conventions {

    /// The right-multiply that flips the camera y and z axes (OpenGL → OpenCV).
    private static let glToCV = simd_float4x4(diagonal: SIMD4<Float>(1, -1, -1, 1))

    /// Convert an ARKit `camera.transform` (camera-to-world) to an OpenCV
    /// camera-to-world pose, returned as a row-major 3×3 rotation and a
    /// 3-vector translation, ready to serialize into `poses.json`.
    static func openCVCameraToWorld(from transform: simd_float4x4) -> (R: [[Double]], t: [Double]) {
        let m = transform * glToCV

        // Row-major R: R[row][col] = m.columns[col][row].
        let R: [[Double]] = [
            [Double(m.columns.0.x), Double(m.columns.1.x), Double(m.columns.2.x)],
            [Double(m.columns.0.y), Double(m.columns.1.y), Double(m.columns.2.y)],
            [Double(m.columns.0.z), Double(m.columns.1.z), Double(m.columns.2.z)],
        ]
        // Translation is unchanged by the axis flip.
        let t: [Double] = [
            Double(transform.columns.3.x),
            Double(transform.columns.3.y),
            Double(transform.columns.3.z),
        ]
        return (R, t)
    }
}

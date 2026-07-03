import Foundation
import UIKit
import CoreGraphics

/// Lossless PNG encoders. Called on the writer queue (off the AR delegate
/// queue). `rgb/000001.png` is lossless per `io_contracts/capture_session.md`;
/// `confidence/000001.png` is an 8-bit grayscale validity mask (255/0).
enum PngWriter {

    /// Lossless PNG of an already-decoded RGB `CGImage`.
    static func pngRGB(_ cg: CGImage) -> Data? {
        UIImage(cgImage: cg).pngData()
    }

    /// Lossless 8-bit grayscale PNG from a `[UInt8]` mask (row-major [H,W]).
    static func pngGray(_ bytes: [UInt8], w: Int, h: Int) -> Data? {
        guard bytes.count == w * h,
              let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        guard let cg = CGImage(width: w, height: h,
                               bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: w,
                               space: CGColorSpaceCreateDeviceGray(),
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                               provider: provider, decode: nil,
                               shouldInterpolate: false, intent: .defaultIntent) else { return nil }
        return UIImage(cgImage: cg).pngData()
    }
}

import Foundation

/// Writes NumPy `.npy` v1.0 files (little-endian float32, C order) so the depth
/// maps this app produces load directly with `numpy.load` in the pipeline,
/// exactly matching `io_contracts/capture_session.md` (`depth/000001.npy`,
/// float32 `[H, W]`, meters, NaN = invalid).
///
/// Format (NEP .npy v1.0): the magic string `\x93NUMPY`, version bytes `1 0`,
/// a little-endian uint16 header length, then an ASCII header dict terminated by
/// `\n` and space-padded so that `10 + headerLen` is a multiple of 64, followed
/// by the raw little-endian float32 payload. ARM64 (the iPhone) is little-endian,
/// so `Float` bit patterns are already `<f4`.
enum NumpyWriter {

    static func npyData(_ values: [Float], shape: [Int]) -> Data {
        var data = header(shape: shape)
        values.withUnsafeBufferPointer { buf in
            data.append(Data(buffer: buf))   // little-endian float32, C order
        }
        return data
    }

    static func write(_ values: [Float], shape: [Int], to url: URL) throws {
        let expected = shape.reduce(1, *)
        precondition(values.count == expected,
                     "NumpyWriter: \(values.count) values != product of shape \(shape) = \(expected)")
        try npyData(values, shape: shape).write(to: url, options: .atomic)
    }

    // MARK: - header

    private static func header(shape: [Int]) -> Data {
        let shapeStr: String
        if shape.count == 1 {
            shapeStr = "(\(shape[0]),)"
        } else {
            shapeStr = "(" + shape.map(String.init).joined(separator: ", ") + ")"
        }
        var dict = "{'descr': '<f4', 'fortran_order': False, 'shape': \(shapeStr), }"

        // 10 = 6 (magic) + 2 (version) + 2 (uint16 length). Pad the dict with
        // spaces and a trailing newline so (10 + headerLen) % 64 == 0.
        let base = 10 + dict.count + 1                 // +1 for the terminating '\n'
        let pad = (64 - base % 64) % 64
        dict += String(repeating: " ", count: pad) + "\n"

        var out = Data()
        out.append(0x93)
        out.append(contentsOf: Array("NUMPY".utf8))
        out.append(contentsOf: [0x01, 0x00])           // version 1.0
        let len = UInt16(dict.utf8.count)
        out.append(UInt8(len & 0xff))                  // little-endian uint16
        out.append(UInt8((len >> 8) & 0xff))
        out.append(contentsOf: Array(dict.utf8))
        return out
    }
}

import Foundation

/// Writes the exact `capture/` output contract (io_contracts/capture_session.md)
/// to disk. Frame files are appended off the AR delegate queue on a private
/// serial queue; `finalize` runs after all appends (serial FIFO) and writes the
/// JSON + README. `@unchecked Sendable`: all mutable state is confined to `io`.
final class FrameWriter: @unchecked Sendable {

    struct Result: Sendable {
        let captureDir: URL
        let frameCount: Int
        let errors: [String]
    }

    let captureDir: URL
    private let rgbDir: URL
    private let depthDir: URL
    private let confDir: URL
    private let io = DispatchQueue(label: "capture.writer.io", qos: .userInitiated)

    // Serial-queue-confined accumulators.
    private var posesJSON: [String: [String: Any]] = [:]
    private var timestamps: [String: Double] = [:]
    private var intrinsicsDoc: [String: Any]?
    private var frameCount = 0
    private var errors: [String] = []

    init(captureDir: URL) throws {
        self.captureDir = captureDir
        self.rgbDir = captureDir.appendingPathComponent("rgb")
        self.depthDir = captureDir.appendingPathComponent("depth")
        self.confDir = captureDir.appendingPathComponent("confidence")
        let fm = FileManager.default
        for d in [captureDir, rgbDir, depthDir, confDir] {
            try fm.createDirectory(at: d, withIntermediateDirectories: true)
        }
    }

    /// Fire-and-forget append; ordering across frames does not matter (each
    /// frame writes its own indexed files and keyed metadata).
    func append(_ p: FramePayload) {
        io.async { self.writeFrame(p) }
    }

    /// All-or-nothing per frame: an index ends up with rgb + depth + confidence
    /// files AND its pose/timestamp/count, or with none of them. This preserves
    /// the count-parity invariant (#rgb == #depth == #confidence == #poses ==
    /// #timestamps) even if an encode or disk write fails. Skipped indices leave
    /// a gap in the 6-digit sequence, which is fine — every stage matches frames
    /// by id (intersection of the ids present), never by assuming 1..N.
    private func writeFrame(_ p: FramePayload) {
        let name = String(format: "%06d", p.index)

        guard p.depth.count == p.depthW * p.depthH else {
            errors.append("frame \(name): depth count \(p.depth.count) != \(p.depthW)x\(p.depthH); dropped")
            return
        }
        // Encode all three first; a nil means we write nothing for this index.
        guard let rgbData = PngWriter.pngRGB(p.color) else {
            errors.append("frame \(name): RGB PNG encode failed; dropped for parity"); return
        }
        let depthData = NumpyWriter.npyData(p.depth, shape: [p.depthH, p.depthW])
        guard let confData = PngWriter.pngGray(p.mask, w: p.depthW, h: p.depthH) else {
            errors.append("frame \(name): confidence PNG encode failed; dropped for parity"); return
        }

        let rgbURL = rgbDir.appendingPathComponent("\(name).png")
        let depthURL = depthDir.appendingPathComponent("\(name).npy")
        let confURL = confDir.appendingPathComponent("\(name).png")
        do {
            try rgbData.write(to: rgbURL, options: .atomic)
            try depthData.write(to: depthURL, options: .atomic)
            try confData.write(to: confURL, options: .atomic)
        } catch {
            errors.append("frame \(name): write failed (\(error.localizedDescription)); dropped for parity")
            for u in [rgbURL, depthURL, confURL] { try? FileManager.default.removeItem(at: u) }
            return
        }

        // All three files are on disk → now (and only now) record the metadata.
        // Pose is optional: the AVFoundation (HQ-Depth) path carries none, so poses.json
        // is omitted entirely and the Linux pipeline recovers pose via unseeded SfM.
        if let R = p.R, let t = p.t {
            posesJSON[name] = ["R": R, "t": t]
        }
        timestamps[name] = p.time
        if intrinsicsDoc == nil {
            intrinsicsDoc = [
                "convention": "OpenCV",
                "color_resolution": [p.colorW, p.colorH],
                "depth_resolution": [p.depthW, p.depthH],
                "intrinsic_matrix_applies_to": "color",
                "K": p.K,
            ]
        }
        frameCount += 1
    }

    /// Write intrinsics.json, poses.json (only if a pose stream exists), timestamps.json,
    /// metadata.json, README, and return a summary. Runs after all pending appends (serial FIFO).
    /// `trackingWasNormalThroughout` / `confidenceNote` are ARKit-specific (pass nil for HQ-Depth).
    func finalize(metadata: CaptureMetadata,
                  trackingWasNormalThroughout: Bool?,
                  confidenceNote: String) async -> Result {
        await withCheckedContinuation { cont in
            io.async {
                if let intr = self.intrinsicsDoc {
                    self.writeJSON(intr, to: self.captureDir.appendingPathComponent("intrinsics.json"))
                }
                // Pose stream is present only for ARKit; omit poses.json entirely otherwise so the
                // pipeline knows to run unseeded SfM (metadata.provides_pose = false says the same).
                let hasPoses = !self.posesJSON.isEmpty
                if hasPoses {
                    self.writeJSON(["convention": "OpenCV",
                                    "pose_type": "camera_to_world",
                                    "poses": self.posesJSON],
                                   to: self.captureDir.appendingPathComponent("poses.json"))
                }
                self.writeJSON(["unit": "seconds", "timestamps": self.timestamps],
                               to: self.captureDir.appendingPathComponent("timestamps.json"))

                var metaDoc = metadata.dictionary()
                metaDoc["frame_count"] = self.frameCount          // authoritative post-write count
                metaDoc["has_poses"] = hasPoses
                self.writeJSON(metaDoc, to: self.captureDir.appendingPathComponent("metadata.json"))

                let colorRes = (self.intrinsicsDoc?["color_resolution"] as? [Int]) ?? [0, 0]
                let depthRes = (self.intrinsicsDoc?["depth_resolution"] as? [Int]) ?? [0, 0]
                let trackingLine = trackingWasNormalThroughout.map {
                    "Tracking stayed normal throughout: \($0 ? "yes" : "NO — inspect before trusting").\n"
                } ?? ""
                let readme = """
                Capture session: \(metadata.sessionID)
                Description: \(metadata.description.isEmpty ? "(none)" : metadata.description)
                Framework: \(metadata.backend.rawValue)   Orientation(framing): \(metadata.orientation.rawValue)
                Coordinate convention: OpenCV (camera looks down +z, x right, y down).
                Units: meters, float32.
                color_resolution: \(colorRes[0])x\(colorRes[1])   (rgb/*.png, lossless)
                depth_resolution: \(depthRes[0])x\(depthRes[1])   (depth/*.npy, float32 [H,W], NaN = invalid)
                Pose stream present: \(hasPoses ? "yes (poses.json, camera_to_world, metric)" : "NO — recover pose via unseeded SfM").
                Depth source: \(metadata.depthSource).
                Confidence: \(confidenceNote)
                World frame: \(hasPoses ? "gravity-aligned (Y up); world yaw arbitrary at start" : "n/a (no live tracking)").
                \(trackingLine)Frames: \(self.frameCount).
                Produced by AnatomyCapture (Stage 1). See io_contracts/capture_session.md.
                """
                try? readme.data(using: .utf8)?.write(
                    to: self.captureDir.appendingPathComponent("README"), options: .atomic)

                cont.resume(returning: Result(captureDir: self.captureDir,
                                              frameCount: self.frameCount,
                                              errors: self.errors))
            }
        }
    }

    // MARK: - helpers

    private func writeJSON(_ obj: [String: Any], to url: URL) {
        do {
            let data = try JSONSerialization.data(withJSONObject: obj,
                                                  options: [.prettyPrinted, .sortedKeys])
            try data.write(to: url, options: .atomic)
        } catch {
            errors.append("\(url.lastPathComponent): \(error.localizedDescription)")
        }
    }
}

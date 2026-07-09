import Foundation
import simd

/// Capture tuning read from UserDefaults (settable in SettingsView), with defaults sized for the
/// pipeline's STRONG-CAPTURE branch (docs/PIPELINE_RECOMMENDATION.md: 150-400 sharp frames). The
/// shipped 60-frame / 20 s cap only ever produced weak captures. Read fresh at each recording so a
/// Settings change takes effect without an app restart. UserDefaults is thread-safe (read off the
/// AR delegate queue is fine).
enum CaptureTuning {
    static var maxKeyframes: Int {
        let v = UserDefaults.standard.integer(forKey: "max_keyframes")
        return v > 0 ? min(max(v, 30), 1000) : 360
    }
    static var budgetSeconds: TimeInterval {
        let v = UserDefaults.standard.double(forKey: "budget_seconds")
        return v > 0 ? min(max(v, 10), 300) : 120
    }
}

/// Selects keyframes by camera motion: a new keyframe roughly every 7.5° of orbital motion, with a
/// uniform time-stride fallback so frames still fill the time budget if the camera stalls. Cap +
/// budget come from `CaptureTuning` (defaults 360 frames / 120 s).
///
/// Confined to the AR delegate serial queue (see `SessionCoordinator`); not
/// thread-safe on its own.
struct KeyframeSelector {
    let angleGateRadians: Float = 7.5 * .pi / 180        // ~0.1309 rad
    let maxKeyframes: Int
    let budgetSeconds: TimeInterval
    let timeStride: TimeInterval

    /// Reads current tuning at construction. `timeStride` targets ~80% of the cap by time so the
    /// angle-gate can add the remainder on a good orbit.
    init(maxKeyframes: Int = CaptureTuning.maxKeyframes,
         budgetSeconds: TimeInterval = CaptureTuning.budgetSeconds) {
        self.maxKeyframes = maxKeyframes
        self.budgetSeconds = budgetSeconds
        self.timeStride = budgetSeconds / Double(max(Int(Double(maxKeyframes) * 0.8), 1))
    }

    private(set) var kept: Int = 0
    private var lastRot: simd_float3x3?
    private var lastKeptTime: TimeInterval?
    private var startTime: TimeInterval?

    mutating func reset() {
        kept = 0
        lastRot = nil
        lastKeptTime = nil
        startTime = nil
    }

    /// Elapsed seconds since the first frame seen (nil before the first call).
    func elapsed(now: TimeInterval) -> TimeInterval {
        guard let s = startTime else { return 0 }
        return now - s
    }

    /// True when the recording should stop (budget elapsed or cap reached).
    func isFinished(now: TimeInterval) -> Bool {
        kept >= maxKeyframes || (startTime.map { now - $0 >= budgetSeconds } ?? false)
    }

    /// Decide whether to keep this frame. `rotation` is the ARKit camera_to_world
    /// rotation (R_arkit); the angular test is convention-independent.
    mutating func shouldKeep(rotation: simd_float3x3, time: TimeInterval) -> Bool {
        if startTime == nil { startTime = time }
        if time - startTime! >= budgetSeconds || kept >= maxKeyframes { return false }

        guard let last = lastRot, let lastT = lastKeptTime else {
            lastRot = rotation; lastKeptTime = time; kept += 1
            return true                                   // always keep the first
        }

        // relative rotation angle: theta = acos((trace(lastᵀ·R) - 1) / 2)
        let rel = last.transpose * rotation
        let trace = rel.columns.0.x + rel.columns.1.y + rel.columns.2.z
        let cosTheta = max(-1, min(1, (trace - 1) / 2))
        let theta = acos(cosTheta)

        let byAngle = theta >= angleGateRadians
        let byTime = (time - lastT) >= timeStride
        if byAngle || byTime {
            lastRot = rotation; lastKeptTime = time; kept += 1
            return true
        }
        return false
    }
}

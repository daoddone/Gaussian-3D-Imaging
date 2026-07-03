import Foundation
import simd

/// Selects keyframes by camera motion, per the build spec: a new keyframe
/// roughly every 7.5° of orbital motion, with a uniform time-stride fallback so
/// frames still fill the 20 s budget if the camera stalls. Target ~48, hard cap
/// 60, hard stop at 20 s.
///
/// Confined to the AR delegate serial queue (see `SessionCoordinator`); not
/// thread-safe on its own.
struct KeyframeSelector {
    let angleGateRadians: Float = 7.5 * .pi / 180        // ~0.1309 rad
    let timeStride: TimeInterval = 20.0 / 48             // ~0.417 s
    let maxKeyframes: Int = 60
    let budgetSeconds: TimeInterval = 20

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

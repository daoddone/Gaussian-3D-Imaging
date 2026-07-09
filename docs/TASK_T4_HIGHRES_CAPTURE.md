# TASK T4 — high-resolution capture modes + LiDAR toggle (iOS capture app)

**Status: ACTIVE. Fable-safe (pure phone camera/sensor code, domain-neutral). Owner is in the loop for
on-device testing.** This spec was produced by a multi-agent read of the current source (line numbers
verified) + adversarial verification. It supersedes any "deferred" framing.

**Scope note — the frame-cap half is ALREADY DONE.** The verified plan had two parts: (A) capture-mode
matrix, (B) raise the frame cap. **(B) is shipped** via `CaptureTuning` (UserDefaults `max_keyframes`
/`budget_seconds`, defaults 360/120) + Settings steppers + per-recording selector re-creation in both
`SessionCoordinator.startWriting` and `AVFoundationCaptureSource.startWriting`. **T4 = part (A) only.**

All paths under `stages/stage1_capture/AnatomyCapture/`.

## The goal
Add a user-selectable **capture-mode matrix**: `{ARKit-1080, ARKit-4K, HQ-stills} × {LiDAR on/off}`,
recorded into metadata, so we can A/B capture resolution + sensor config. VIO+LiDAR concurrent recording
already exists (the ARKit backend); this adds **true high resolution** and an explicit LiDAR gate.

## ⚠️ The load-bearing correctness point (do NOT ship the shortcut)
On a LiDAR iPhone Pro, the default ARKit `videoFormat` AND
`recommendedVideoFormatForHighResolutionFrameCapturing` are **both ~1920×1440@60**. So merely setting a
"4K" videoFormat yields data IDENTICAL to 1080/1440 — a capture-day "1080 vs 4K" A/B would compare two
identical configs. **The only real resolution gain is the 12 MP still via
`ARSession.captureHighResolutionFrame(completion:)`** (iOS 16+), pulled on the keyframe gate WHILE the
1440p stream + sceneDepth keep running (verified: depth via `frame.sceneDepth.depthMap`, uninterrupted).
**ARKit-4K MUST deliver the 12 MP still, not just a stream format** — that async still is the real task.

## Implementation (part A)

**1) `CaptureMetadata.swift` — mode enum + metadata.**
- Add `enum CaptureMode: String, CaseIterable, Sendable { case arkit1080, arkit4K, hqStills }` with
  `var uiLabel` and computed `var backend: CaptureBackend { self == .hqStills ? .hqDepth : .arkit }`.
  KEEP the existing `CaptureBackend` (low-level camera owner); all preview/UI branches keep reading
  `model.backend`.
- In `struct CaptureMetadata`: add `captureMode: CaptureMode`, `lidarEnabled: Bool`; extend
  `dictionary()` with `"capture_mode"` + `"lidar_enabled"`.

**2) `CaptureModel.swift` — state + config.**
- Replace `var backend = .arkit` (line ~40) with `var captureMode: CaptureMode = .arkit1080` +
  `var lidarEnabled: Bool = true`; add DERIVED `var backend: CaptureBackend { captureMode.backend }` so
  all existing reads keep working.
- Replace `setBackend(_:)` with `setMode(_:)` (same idle/previewing guard; same outgoing-teardown when
  `m.backend != captureMode.backend`; if only the ARKit resolution changed, set mode + `runConfiguration()`).
  Add `setLidarEnabled(_:)` (same guard; re-run config).
- `makeConfiguration()`: gate `frameSemantics` (133-139) AND `sceneReconstruction` (144-146) behind
  `if lidarEnabled`. For `arkit4K`, set `videoFormat = recommendedVideoFormatForHighResolutionFrameCapturing`
  (keeps 60fps + per-frame sceneDepth). Do NOT use `recommendedVideoFormatFor4KResolution` (30fps,
  undocumented-with-depth).
- **The 12 MP still (the real gain):** on the keyframe gate in `arkit4K`, call
  `session.captureHighResolutionFrame` and write THAT frame's `capturedImage` (+ its `frame.sceneDepth`)
  instead of the streamed frame. This is one async hop inside the delegate — the one non-trivial piece.

**3) `SessionCoordinator.swift` — LiDAR-off + high-res still.**
- Add a `lidarEnabled` flag (set from `CaptureModel` like `depthMode`). When OFF, the depth guard at
  line ~107 drops every frame → instead skip the depth block and append RGB+pose+K with **depth=all-NaN
  sized EXACTLY depthW*depthH** + mask=all-255 (else `FrameWriter` guard at :58 `depth.count==depthW*depthH`
  drops the frame). `providesPose` MUST stay TRUE (VIO pose survives without LiDAR; false wrongly triggers
  unseeded SfM server-side). Set `depth_source="none (RGB only)"` at the construction site in
  `stopRecording` (262-264).

**4) `AVFoundationCaptureSource.swift` — HQ-stills.**
- Add `AVCapturePhotoOutput` (`isDepthDataDeliveryEnabled=true`) in `configureIfNeeded()`, triggered on
  the keyframe gate, reusing the K-scaling + `FramePayload` append. Add `lidarEnabled` (skip depth when off).
- ⚠️ RISK: streaming-depth + photo-depth on ONE `AVCaptureSession` (`.inputPriority`, hand-locked
  `activeFormat`) may fail `canAddOutput()` or be bounded by the locked video format (not sensor-max
  photo). Validate on device; if it conflicts, fall back to periodic still-only capture or keep HQ as the
  existing max-depth video stream.

**5) `CaptureView.swift` — controls.**
- `setupControls`: change the segmented Picker to `CaptureMode.allCases` bound `get:{model.captureMode}
  set:{model.setMode($0)}`; add `Toggle("LiDAR depth", isOn: ...setLidarEnabled)`. topBar branches keep
  reading `model.backend` (derived) unchanged.

## Concurrency invariant (Swift 6 strict)
Any values read from the nonisolated delegate/data queues via the weak `@MainActor CaptureModel` must be
`let` (Sendable). `captureMode`/`lidarEnabled` are read on the MainActor at config time and threaded to
the coordinators as plain flags — keep the cross-queue reads to Sendable copies.

## On-device verification checklist (owner-in-the-loop — cannot check off-hardware)
1. **ARKit-4K really delivers depth:** `session.currentFrame?.sceneDepth != nil` across frames while the
   high-res format + 12 MP still path are active.
2. **12 MP still fires + writes:** confirm `capture/rgb/*.png` for `arkit4K` are ~4032×3024, not 1920×1440
   (this is the A/B-validity check — if they're 1440p, the still path isn't wired).
3. **HQ-stills coexistence:** does `AVCapturePhotoOutput` + depth stream co-exist on the one session, and
   what resolution does the still actually reach?
4. **LiDAR-off:** RGB frames still save; `metadata.json` has `lidar_enabled:false`, `provides_pose:true`,
   `depth_source:"none (RGB only)"`; server treats it as VIO-scale-only.
5. Run the IOS_NOTES §7 orientation self-test on a near-planar frame for EACH new format before trusting
   metric output (sensor-native buffer orientation is the historic #1 bug).

## Why it matters (domain-neutral)
Higher capture resolution sharpens the splat, SfM feature matching, and the texture bake (mesh geometry
is capped at 2048 px, so this is a splat/appearance gain, not a mesh gain). It also gives us a clean
capture-mode A/B for the benchmark (T5). LiDAR-off enables the markerless VIO-only scale experiment
directly from the app (also testable server-side by ignoring depth).

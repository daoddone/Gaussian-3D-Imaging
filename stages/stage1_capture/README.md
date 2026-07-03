# Stage 1 — AnatomyCapture (iPhone LiDAR capture app)

SwiftUI + ARKit app that records a short patient orbit with the rear LiDAR and
writes the exact `capture/` output contract
([`io_contracts/capture_session.md`](../../io_contracts/capture_session.md)):
synchronized color, metric depth, a validity mask, camera intrinsics, a metric
camera path, and timestamps. Target device: **iPhone 14 Pro, iOS 26**, built in
Xcode 26 on a Mac.

- Engineering reference (APIs, conventions, gotchas): [`IOS_NOTES.md`](IOS_NOTES.md).
- Original brief: [`CAPTURE_SPEC.md`](CAPTURE_SPEC.md) — but see the architecture
  note below; its two-stream design is **superseded**.

## Architecture (why one ARKit session, not two streams)

`CAPTURE_SPEC.md` describes running AVFoundation (color+depth) and ARKit (pose)
simultaneously. **That is not runnable:** an `AVCaptureSession` and an `ARSession`
cannot share the rear camera — ARKit takes exclusive control (Apple staff, Dev
Forums 81971; WWDC22). Because a metric `poses.json` is mandatory and only ARKit
produces a drift-corrected metric pose, the app uses a **single
`ARWorldTrackingConfiguration` session** and reads everything from one `ARFrame`:

| Contract file | ARKit source | Notes |
| --- | --- | --- |
| `rgb/*.png` | `frame.capturedImage` (1920×1440 YCbCr → RGB) | lossless PNG |
| `depth/*.npy` | `frame.smoothedSceneDepth.depthMap` (256×192 Float32 m) | NaN where invalid |
| `confidence/*.png` | `…confidenceMap` (ARConfidenceLevel) | 255 where ≥ `.medium`, else 0 |
| `intrinsics.json` | `frame.camera.intrinsics` @ color res | depth K = ×(256/1920) |
| `poses.json` | `frame.camera.transform` → OpenCV `camera_to_world` | `Conventions.swift` flip |
| `timestamps.json` | `frame.timestamp` (relative to first kept frame) | seconds |

**Tradeoff:** ARKit depth is 256×192 and temporally smoothed (vs AVFoundation's
~320×240 raw), but it comes with the metric pose in one aligned frame. Color
still drives fine detail downstream. Depth is within the contract's "roughly
320×240" tolerance; the actual resolutions are recorded in `intrinsics.json` and
the session `README`.

## Build & run

```bash
brew install xcodegen
cd stages/stage1_capture
xcodegen generate            # AnatomyCapture.xcodeproj (gitignored)
open AnatomyCapture.xcodeproj
```
In Xcode: select **AnatomyCapture** → **Signing & Capabilities** → set your Team,
pick your iPhone 14 Pro as the run destination, and **Run**. Physical device only
(ARKit does not run in the Simulator). No extra dependencies — the app uses only
system frameworks (SwiftUI, ARKit, RealityKit, CoreImage, Accelerate/ImageIO).

> No XcodeGen? Create a new iOS App (SwiftUI) project named `AnatomyCapture`, drag
> in everything under `AnatomyCapture/` (Swift files + `Info.plist` +
> `Assets.xcassets`), set the Info.plist keys listed in `IOS_NOTES.md §1`, and Run.

## Use

1. Point the reticle at the region of interest, ~30 cm away.
2. Tap the red button and **orbit slowly**; finish within ~20 s.
3. The app keeps a keyframe roughly every 7.5° of motion (target ~48, hard cap
   60) and auto-stops at 20 s / 60 frames. Live readouts: frame count, elapsed,
   valid-depth %, tracking state.
4. On finish it writes `Documents/sessions/<id>/capture/…` and offers a **Share
   .zip** (AirDrop / Save to Files). The folder is also in **Files → On My iPhone
   → AnatomyCapture → sessions/**.

## Get the capture into the pipeline

AirDrop / copy the session folder to the machine, then place it so the pipeline
sees `sessions/<id>/capture/…`. Verify orientation/scale with the pipeline's
orientation self-test before trusting results:

```bash
python -m common.orientation_selftest
# then Stage 2 (frontend) → Stage 3 (metric) on sessions/<id>
```

## Files

| File | Role |
| --- | --- |
| `AnatomyCaptureApp.swift` | `@main` SwiftUI app. |
| `CaptureView.swift` | ARView preview, orbit reticle, record/stop, status, share. |
| `CaptureModel.swift` | `@MainActor @Observable` lifecycle; owns config + finalize. |
| `SessionCoordinator.swift` | `ARSessionDelegate` (nonisolated); copies buffers, gates keyframes. |
| `KeyframeSelector.swift` | 7.5° angular gate + time-stride fallback + caps. |
| `PixelBufferCopy.swift` | Deep copies: color→CGImage, depth→[Float]+mask. |
| `PngWriter.swift` / `NumpyWriter.swift` | Lossless PNG / `.npy` v1.0 encoders. |
| `Conventions.swift` | ARKit→OpenCV `camera_to_world` (verified vs Python `common/`). |
| `FrameWriter.swift` | Writes the exact contract folder + JSON + README. |
| `Exporter.swift` | Zips the session for sharing. |
| `Info.plist`, `project.yml`, `Assets.xcassets` | App config / XcodeGen spec / assets. |

## To verify on device (from `IOS_NOTES.md §14`)

`supportsFrameSemantics(.smoothedSceneDepth)` true; `camera.imageResolution`
1920×1440 and depthMap 256×192; `confidenceMap` non-nil; depth stays registered
to color; **run the orientation self-test on a real capture to confirm the R
sign convention**; no dropped-frame warnings under the copy+encode load.

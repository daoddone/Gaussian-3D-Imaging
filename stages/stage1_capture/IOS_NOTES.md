# Stage 1 Capture — iOS Engineering Notes (IOS_NOTES.md)

Target: iPhone 14 Pro, iOS 26.5.2, built in Xcode 26 on a Mac. Produces the exact `capture/` contract in `io_contracts/capture_session.md`. This document is the build reference; it supersedes the two-stream design in `CAPTURE_SPEC.md` (see Architecture).

## 0. Architecture decision (read first)

**Ship the ARKit-unified path — a single `ARSession`.** You CANNOT run an `AVCaptureSession` and an `ARSession` on the rear camera at the same time; ARKit takes exclusive control of the capture device (Apple staff, Dev Forums 81971; WWDC22 "…ARKit does not support using the camera while multitasking"). The only sanctioned device touch during ARKit is the read-only `ARConfiguration.configurableCaptureDeviceForPrimaryCamera`.

Because `poses.json` (metric `camera_to_world`) is mandatory and AVFoundation LiDAR gives **no** pose, and the two frameworks cannot coexist, the unified ARKit path is the only runnable architecture that produces the full contract. **The `CAPTURE_SPEC.md` "Stream A (AVFoundation) + Stream B (ARKit) aligned by timestamp" design is not runnable and is not built.**

Tradeoff (state in the session README): ARKit depth is 256x192, Apple-processed / temporally smoothed, not raw; AVFoundation could give ~320x240 (up to 768x576 photo) but pose-less. We accept lower/filtered depth to get the metric pose in one aligned frame. ARKit depth never emits NaN — the app synthesizes NaN-invalid depth and the 255/0 mask from `confidenceMap`.

Consequences for the contract numbers: `depth_resolution = [256,192]`, `color_resolution = [1920,1440]`, `poses.json` **always present** (the "omit poses" fallback is void).

## 1. Project setup

- Generate the project with **XcodeGen** (`brew install xcodegen; xcodegen generate`); check in `project.yml`, gitignore `*.xcodeproj`. Do not hand-author `project.pbxproj` (objectVersion 77, one bad 24-hex UUID corrupts it).
- `project.yml`: `deploymentTarget.iOS: "26.0"`, `SWIFT_VERSION: "6.0"`, target type `application`, `TARGETED_DEVICE_FAMILY: "1"` (iPhone), `GENERATE_INFOPLIST_FILE: NO` (supply Info.plist), `DEVELOPMENT_TEAM` = your team id.
- Build with the iOS 26 SDK (Xcode 26+). Physical device only — ARKit does not run in the simulator.

### Info.plist (required)
- `NSCameraUsageDescription` (String) — required or the app crashes on session start.
- `UIRequiredDeviceCapabilities` = `["arkit"]` — gates install to LiDAR/A12+ devices.
- `UISupportedInterfaceOrientations` = `["UIInterfaceOrientationLandscapeRight"]` — **lock to a single landscape orientation** (see §6, intrinsics are orientation-dependent on iPhone). Also override `supportedInterfaceOrientations = .landscapeRight` in the scene/host.
- `UIFileSharingEnabled` = YES and `LSSupportsOpeningDocumentsInPlace` = YES — both, so the session folder appears and is editable in the Files app (On My iPhone).

## 2. Swift 6.2 / Xcode 26 concurrency (this WILL bite the delegate)

New Xcode 26 app templates turn on `SWIFT_APPROACHABLE_CONCURRENCY=YES` and `SWIFT_DEFAULT_ACTOR_ISOLATION=MainActor` — **all your code is implicitly `@MainActor` unless marked `nonisolated`**. `ARSessionDelegate` callbacks arrive on the background `delegateQueue`, so:

- Mark the delegate method `nonisolated func session(_:didUpdate:)` (or make the whole `SessionCoordinator` nonisolated). Inside it you cannot touch `@MainActor` UI state directly — hop with `Task { @MainActor in model.note(...) }`.
- `ARFrame` and `CVPixelBuffer` are **not Sendable**. Never retain the frame or pass a buffer across an isolation hop. Copy out only value types: `[Float]`, `[UInt8]`, `simd_float3x3`, `simd_float4x4`, `TimeInterval`, `CGImage`.
- Do disk I/O in a serial `actor FrameWriter`. If a `nonisolated async` write should run off any actor use `@concurrent`; a single actor is enough for <=60 frames. SE-0461 (`nonisolated(nonsending)`) means nonisolated async fns inherit the caller's actor by default — be explicit.

## 3. Session configuration & lifecycle

```swift
let cfg = ARWorldTrackingConfiguration()               // worldAlignment defaults to .gravity (Y up, gravity-aligned, origin at run())
precondition(ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth))
cfg.frameSemantics = [.smoothedSceneDepth]             // temporally denoised — best for a slow orbit; use [.sceneDepth] for raw
// keep the DEFAULT videoFormat so depth stays registered to color; do NOT switch to a hi-res video format
session.delegate = coordinator
session.delegateQueue = DispatchQueue(label: "ar.delegate", qos: .userInitiated)
session.run(cfg, options: [.resetTracking, .removeExistingAnchors])
```
- Do NOT use `captureHighResolutionFrame` or `recommendedVideoFormatForHighResolutionFrameCapturing` for the stream: the 12MP still is out-of-band, async, and can **de-register** sceneDepth from color. Accept 1920x1440 as "highest practical" for an aligned RGB-D stream.
- Keep `worldAlignment = .gravity` (not `.gravityAndHeading` — adds startup latency). Document that world yaw is arbitrary at session start.
- Only accept keyframes when `frame.camera.trackingState == .normal`; skip `.limited`/`.notAvailable`.
- Stop at 20 s or 60 keyframes: `session.pause()` then finalize JSON.

## 4. Per-frame extraction (resolutions & formats, iPhone 14 Pro)

| Stream | Property | Res (WxH) | Pixel format | Notes |
|---|---|---|---|---|
| Color | `frame.capturedImage` | 1920x1440 (4:3) | `420YpCbCr8BiPlanarFullRange` ('420f') | biplanar YCbCr, FULL range, NOT RGB — must convert |
| Depth | `frame.smoothedSceneDepth.depthMap` | 256x192 | `DepthFloat32` | Float32 meters along camera ray |
| Confidence | `…confidenceMap` | 256x192 | `OneComponent8` ('L008') | byte = ARConfidenceLevel 0/1/2 |

**Always read sizes at runtime** (`CVPixelBufferGetWidth/Height`, `frame.camera.imageResolution`) — do not hardcode; some builds expose 1440x1080 color. Copy discipline:
- `CVPixelBufferLockBaseAddress(buf, .readOnly)` … `Unlock(…, .readOnly)`.
- Stride by `CVPixelBufferGetBytesPerRow` (padding: rowBytes may exceed width*elem). Depth/confidence are single-plane; color is bi-planar (`…BaseAddressOfPlane`).
- memcpy into your own arrays synchronously, unlock, return. Retaining ARFrames stalls the reuse pool ("dropping frame" warnings).

## 5. Depth -> NaN and confidence mask (synthesized)

ARKit depth is dense (no NaN); validity comes from `confidenceMap`. Per contract:
```
valid  = depth.isFinite && depth > 0 && conf >= 1        // >= .medium
depth  = valid ? depth : Float.nan                        // depth/000001.npy
mask   = valid ? 255 : 0                                   // confidence/000001.png (8-bit gray 256x192)
```
Document the threshold (`>= .medium`) in the README. `depth/000001.npy` is float32 shape `[192,256]` = `[H,W]`, C-order, `<f4`, via existing `NumpyWriter.write(depth, shape:[192,256], to:)`.

## 6. Orientation & intrinsics (the #1 pipeline bug)

- On iPhone (unlike iPad) `camera.intrinsics`/`imageResolution` are reported for the **sensor-native landscape** frame and CHANGE with interface orientation. Lock the app to `landscapeRight` (Info.plist + scene override). Save all buffers un-rotated (no `displayTransform`, no CGImagePropertyOrientation). Then color, depth, confidence, K, and transform share one frame.
- `intrinsics` (simd, column-major): `fx = K.columns.0.x`, `fy = K.columns.1.y`, `cx = K.columns.2.x`, `cy = K.columns.2.y`. K applies to the **color** `imageResolution` (1920x1440), NOT to depth.
- `intrinsics.json`: store K at color res with `intrinsic_matrix_applies_to = "color"`, `color_resolution [1920,1440]`, `depth_resolution [256,192]`. A consumer scales K to depth by the uniform factor `256/1920 = 192/1440 = 0.13333` (`fx'=fx*s, cx'=cx*s, fy'=fy*s, cy'=cy*s`). Because color and depth share 4:3 aspect and are both sensor-native, this single scale is exact.

## 7. Pose: ARKit -> OpenCV (already implemented, verified)

`Conventions.openCVCameraToWorld(from:)` is correct and should be used as-is:
- `frame.camera.transform` is `camera_to_world`, simd column-major, meters. ARKit camera = OpenGL-style (+X right, +Y up, -Z forward). OpenCV = +X right, +Y down, +Z forward.
- `R_opencv = R_arkit * diag(1,-1,-1)` (negate the Y and Z basis columns); `t` unchanged (same physical point in meters). The file computes `transform * diag(1,-1,-1,1)`, reads R row-major `R[row][col]=m.columns[col][row]`, and takes `t` from the original transform. Store R (3x3 row-major) + t (3) under `pose_type: "camera_to_world"`, `convention: "OpenCV"`. Verify signs with the Section 5 orientation self-test before trusting output.

## 8. Color YCbCr -> lossless PNG

- Convert with a single reused Metal `CIContext(mtlDevice: MTLCreateSystemDefaultDevice()!)`. `CIImage(cvPixelBuffer:)` handles the 601 full-range YCbCr->RGB.
- Encode `CIFormat.RGBA8` + `CGColorSpace(name: .sRGB)` via `ctx.pngRepresentation(of:format:colorSpace:)`, or `CGImageDestination` with `UTType.png`. Source is 8-bit YCbCr so RGBA8/sRGB is lossless w.r.t. sensor data. Wrong colorspace/range => washed-out PNG.
- Grayscale confidence PNG: 8-bit single channel, 256x192.

## 9. NPY writer (already implemented)

`NumpyWriter.swift` emits v1.0: magic `\x93NUMPY`, version `1 0`, LE uint16 header len, dict `{'descr': '<f4', 'fortran_order': False, 'shape': (192, 256), }` space-padded so `10+len` %64==0 and `\n`-terminated, then raw LE float32 C-order (arm64 is LE, so `Float` bits == `<f4`). NaN passes through as `0x7FC00000`. Pass `shape:[H,W]=[192,256]`.

## 10. Keyframe selection

- Gate on `trackingState == .normal`.
- Angular gate: relative rotation `theta = acos(clamp((trace(lastKeyRᵀ·R)-1)/2, -1, 1))`; keep when `theta >= 7.5° (0.1309 rad)`. Equivalent: `simd_quatf(R).angle` vs previous.
- Fallback: if rotation stalls, also keep on a uniform **time stride** `20s/48 ≈ 0.417s` so frames fill within budget.
- Hard cap 60, target ~48, hard-stop at 20 s. Indices 1-based, `String(format:"%06d", n)`.

## 11. JSON / README (exact schemas)

Match `io_contracts/capture_session.md`:
- `intrinsics.json`: `{convention:"OpenCV", color_resolution:[1920,1440], depth_resolution:[256,192], intrinsic_matrix_applies_to:"color", K:[[fx,0,cx],[0,fy,cy],[0,0,1]]}`.
- `poses.json`: `{convention:"OpenCV", pose_type:"camera_to_world", poses:{"000001":{R:[[…]], t:[…]}}}`.
- `timestamps.json`: `{unit:"seconds", timestamps:{"000001":0.000, …}}` — store **relative to first kept frame** (`frame.timestamp - t0`; raw timestamp is mach uptime seconds, not epoch).
- `README`: OpenCV convention; color 1920x1440, depth 256x192; pose stream present (yes); depth mode (smoothedSceneDepth); confidence threshold (>= .medium); note world yaw arbitrary/gravity-aligned.
Use `JSONEncoder` with `.sortedKeys`/`.prettyPrinted`. Write files with `FileManager.createDirectory(withIntermediateDirectories:true)`; `.atomic` writes.

## 12. Completion checks (before declaring capture done)

- Count parity: #rgb == #depth == #confidence == #poses == #timestamps.
- Valid-depth fraction reasonable per frame (not overwhelmingly NaN) — catches too-far / non-returning surfaces.
- Tracking stayed `.normal` throughout (no long gaps). Surface these in the UI.

## 13. Export

Session root: `FileManager.default.urls(for:.documentDirectory,…)[0]/sessions/<id>/capture/` — visible in Files via the Info.plist keys. Zip with zero deps: `NSFileCoordinator().coordinate(readingItemAt: dir, options: .forUploading, error:&e){ zipURL in copy to temp }` (the coordinated URL is a .zip). Share via SwiftUI `ShareLink(item: zipURL)` or `UIActivityViewController` (AirDrop / Save to Files).

## 14. On-device verification checklist (Xcode 26)
1. `supportsFrameSemantics(.smoothedSceneDepth) == true` (and `.sceneDepth`).
2. Print `frame.camera.imageResolution` (expect 1920x1440) and depthMap `Get Width/Height` (expect 256x192).
3. Confirm `confidenceMap != nil` and pixel format is `OneComponent8`.
4. Confirm depth stays registered to color at the default videoFormat (no de-registration).
5. Run the orientation self-test (§7) — verify R sign convention and orthonormality.
6. Confirm no dropped-frame warnings under the copy+encode load for ~48 frames in 20 s.
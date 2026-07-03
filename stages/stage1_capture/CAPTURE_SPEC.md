# Stage 1 Capture-Application Data Specification

> **⚠️ Superseded in part.** The **output file contract** below is authoritative and
> unchanged. The **two-stream capture design** (AVFoundation Stream A + ARKit Stream B
> running simultaneously) is **not runnable** — an `AVCaptureSession` and an `ARSession`
> cannot share the rear camera. The shipped app uses a single ARKit session; see
> [`IOS_NOTES.md`](IOS_NOTES.md) and [`README.md`](README.md) for the actual architecture.

> The working specification for the engineer building the iPhone application. It expands
> Stage 1 of the build specification and produces exactly the output file contract defined
> in `io_contracts/capture_session.md`.
>
> Notation: an abbreviation is written out in full the first time with the short form in
> parentheses. Code symbols (for example Apple's `AVCaptureDevice`) and file extensions are
> reproduced exactly.

## Purpose

Build an iPhone application that records a patient capture and writes a session folder containing, for every frame: the color image, the metric depth (distance in meters at each pixel), a depth validity or confidence map, the camera's internal optical parameters, the camera's position and orientation in real-world units, and a timestamp. This replaces the third-party Record3D application so the team owns the capture stack and its data provenance.

## Hardware requirement

An iPhone Pro or iPad Pro that has a rear Light Detection and Ranging (LiDAR) depth sensor. The application uses the rear camera and rear sensor only, because the provider faces the patient.

## The two capture streams and why there are two

Apple exposes the data this project needs across two different frameworks, and each provides something the other does not. The application therefore runs two streams and aligns them by timestamp.

**Stream A, high-quality color and depth, from the direct capture framework (`AVFoundation`).** Use the built-in LiDAR depth capture device, selected as an `AVCaptureDevice` of type `builtInLiDARDepthCamera`. This device fuses the depth sensor with the rear color camera and delivers synchronized color and depth. Use an `AVCaptureDepthDataOutput` to receive the depth as `AVDepthData`, and read the depth values from its depth map, which is a pixel buffer of 32-bit floating-point distances. Two settings matter:

- Request the depth as true distance (depth), not disparity, so the values are in meters.
- Turn depth filtering off (set the output's filtering to disabled). Filtering smooths and fills the depth, which is good for photography but bad for measurement. With filtering off, the device returns the raw depth and marks pixels it is not confident about as invalid (they come back as not-a-number values, meaning "no reliable reading here"). Those invalid pixels are themselves the confidence signal, described under the confidence field below.

This is the higher-quality path: color up to twelve megapixels and depth at a higher resolution than the other framework provides.

**Stream B, the metric camera path, from the augmented-reality framework (`ARKit`).** Run a world-tracking session (`ARWorldTrackingConfiguration`). For each frame, read the camera's position and orientation from the frame's camera transform (`ARCamera.transform`), which is a four-by-four matrix expressed in real-world meters because the tracking fuses the camera with the phone's motion sensors. This stream is the source of the metric camera path, which is one of the independent ways Stage 3 locks true scale. This stream's own depth and images are not used; only the pose is taken from it.

**Why not one stream.** The direct capture framework gives the best color and depth but does not give a drift-corrected world pose. The augmented-reality framework gives the metric pose but lower-resolution depth. Running both and aligning by timestamp gets the best of each. If running both at once proves difficult on the device, see "Fallback" below.

## Resolution and capture-rate decision

There is a genuine tradeoff the engineer must choose between, and it is worth stating plainly:

- Continuous depth (streaming while recording a video-style orbit) arrives at roughly 320 by 240 pixels. This gives dense temporal coverage of the whole surface, which is what reconstruction wants.
- Photograph-style depth capture can reach roughly 768 by 576 pixels, more than double the resolution, but only at individual still moments rather than continuously.

**Recommendation.** Record continuously for coverage, accepting the roughly 320 by 240 depth, because reconstruction benefits more from many overlapping views than from a few high-resolution depth frames. As an optional enhancement, capture a few photograph-quality depth stills at key positions and store them alongside the video-rate frames for the reconstruction to draw on. Color should always be captured at the highest practical resolution, since fine surface detail lives in the color image and the depth sensor cannot resolve it regardless.

**Frame and video budget.** Cap the recorded video at 20 seconds per session. The reconstruction uses a target of 48 selected keyframes (hard maximum 60), chosen by camera motion (a new keyframe roughly every 7.5 degrees of orbital movement) where the pose is available, and by a uniform time stride otherwise. The 20-second cap paces the clinician to complete a smooth orbit while keeping the frame count within what the graphics card can process.

## Coordinate convention and the conversions required

The whole pipeline uses the computer-vision convention (often called OpenCV), in which the camera looks down its positive z axis, with x pointing right and y pointing down. Apple's frameworks do not use this convention, so two conversions are required, and both must be verified with the orientation self-test in Section 5 of the build specification before any output is trusted.

- **Camera pose.** The augmented-reality framework reports the camera looking down its negative z axis with y pointing up (the convention common in graphics). Convert each pose to the computer-vision convention by rotating the camera's own axes by 180 degrees about its x axis, which is equivalent to negating the y and z axes of the camera frame. Store the converted pose.
- **Intrinsics and the depth map.** These are handled below.

## Per-frame data fields and exact formats

Write these to the session folder using the exact names and formats from the file contract. Frame numbering is zero-padded and shared across all fields so that frame `000001` refers to the same instant in every folder.

- **Color image.** `rgb/000001.png`. Lossless format for fidelity. Record its pixel dimensions.
- **Metric depth.** `depth/000001.npy`. A 32-bit floating-point array of shape `[height, width]`, values in meters, invalid pixels stored as not-a-number. This is the raw depth from Stream A with filtering off. Do not upsample it to the color resolution; store it at its native resolution and record that resolution, because a later stage aligns depth to color using the intrinsics.
- **Depth validity or confidence.** `confidence/000001.png`. At minimum, a validity mask: value 255 for "valid reading here" and value 0 for "invalid," derived directly from which depth pixels are not-a-number. This binary mask is sufficient for the pipeline, since Stage 3 and Stage 5 only need to know which depth pixels to trust. If a graded confidence (low, medium, high) is wanted instead, the augmented-reality framework's scene-depth confidence map can supply it at the lower resolution; treat this as an optional upgrade, not a requirement.
- **Camera intrinsics.** `intrinsics.json`. The camera's internal optical parameters: the three-by-three intrinsic matrix (the two focal lengths and the principal point, meaning the image center) plus the image size the matrix applies to. See the intrinsics note below, because this needs care.
- **Camera pose.** `poses.json`. Per-frame position and orientation from Stream B, already converted to the computer-vision convention, in meters.
- **Timestamps.** `timestamps.json`. Per-frame timestamps, used to align the two streams.
- **A `README`** in the session folder stating that all data uses the computer-vision convention and listing the color and depth resolutions.

## The intrinsics note (a common source of error)

Apple reports the camera intrinsic matrix relative to a specific reference resolution, which may differ from the resolution of the image actually saved. The intrinsic matrix must be scaled to match the resolution of the saved color image. Separately, because the depth map is stored at a lower resolution than the color image, record which resolution the stored intrinsics correspond to, and record the depth resolution as well, so a later stage can relate depth pixels to color pixels correctly. Get the intrinsics from the calibration data attached to the captured frames (`AVCameraCalibrationData`), and scale them to the saved image size before writing them.

## Capture technique guidance (affects every downstream stage)

The quality of everything downstream is inherited from this stage, so the application should encourage good technique:

- A slow, steady orbit around the region of interest at a roughly constant distance.
- Enough overlap between consecutive frames, which a slow orbit provides naturally.
- Even, diffuse lighting where possible, avoiding harsh glare on moist tissue.
- Keeping the region of interest centered and filling a good portion of the frame.

Consider a simple on-screen guide (for example, a target reticle and a speed indicator) to help the provider hold distance and pace, and to pace the orbit to finish within the 20-second window. This is provider assistance for capture quality only; it is not a technical decision knob, and it does not change any downstream processing.

## Output folder layout (exact)

```
sessions/<session_id>/capture/
├── README                      # states the coordinate convention and resolutions
├── rgb/000001.png ...
├── depth/000001.npy ...
├── confidence/000001.png ...
├── intrinsics.json
├── poses.json
└── timestamps.json
```

## What the application should check before declaring a capture complete

- That the number of color frames, depth frames, confidence maps, poses, and timestamps all match.
- That a reasonable fraction of depth pixels are valid (not overwhelmingly not-a-number), which catches a capture that was too far away or against a non-returning surface.
- That the pose stream tracked successfully throughout, with no long gaps where tracking was lost.

## Fallback if running both streams at once is difficult

If the device cannot comfortably run the direct capture stream and the tracking stream simultaneously, capture the color and depth from Stream A only, and omit the metric camera path. Stage 3 still has two other independent ways to lock scale, the depth sensor itself and an optional physical ruler in frame, so the pipeline remains metric. Note this choice in the session `README` so Stage 3 knows the camera-path anchor is absent.

## Open decisions for the engineer to record

- Continuous depth only, versus continuous plus occasional photograph-quality depth stills.
- Binary validity mask only, versus adding the graded confidence from the tracking framework.
- Whether both streams run simultaneously, or the fallback is used.

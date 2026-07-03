# Pipeline for "System and Method for Archiving Photorealistic Radiance-Field (View-Dependent 3D Gaussian Splatting) Reconstructions of Patient Anatomy from Video for Health-Record Documentation and Downstream Analysis": Engineering Build Specification

> A build brief for an artificial-intelligence (AI) coding agent and the engineering team.
> This document is intended to be edited stage by stage. Every stage is modular, with a frozen
> input and output contract, so any single component can be swapped later without disturbing the rest.

---

## 0. How to read this document

**Audience.** An AI coding agent that will scaffold and implement this pipeline, plus the human engineers reviewing its work.

**Structure.** The pipeline is a sequence of independent **stages**. Each heavy learned model runs as its own stage, in its own software environment, and communicates only by writing files to disk. The files on disk are the real interface between stages. This is deliberate: these models require mutually incompatible software environments, and file-based hand-off is what lets them cooperate. It is also what makes the pipeline swap-friendly, since replacing a stage only requires that its replacement read and write the same files.

**A note on notation.** Where an abbreviation first appears it is written out in full with the short form in parentheses. File extensions (for example `.ply`, `.json`, `.npy`, `.png`), library names, model identifiers, and code symbols are reproduced verbatim, because they must appear exactly that way in code.

**A note on scope.** This is the **research and publication** pipeline. It selects the single best-quality component at each stage. Commercial licensing is deliberately out of scope here and will be handled separately; some components carry non-commercial research licenses, which is acceptable for publication and validation.

---

## 1. Project purpose and clinical context

This project converts a short smartphone video of a patient into a metrically accurate, photorealistic three-dimensional (3D) surface reconstruction, for clinical documentation, health-record referencing, and longitudinal tracking across visits.

**Capture situation.** A clinical provider holds the phone and records the patient's anatomy in front of them. This includes faces, but also larger body regions and wounds. Because the provider faces the patient, the pipeline uses the **rear** camera and the **rear** depth sensor. The front-facing sensor is not used, even though it is more accurate for faces, because it does not fit the clinical workflow.

**What "good" means here.**

1. **Metric.** The reconstruction must be tied to true physical size, so distances and volumes can be measured. The accuracy target is roughly one millimeter of surface deviation against a gold-standard reference.

2. **Faithful to the abnormal.** The tool exists to document abnormality (wounds, deformities, post-surgical anatomy). No component may impose a model of "healthy" anatomy that could distort an anomaly toward normal. This principle directly determines a component exclusion in Section 12.

3. **Open and inspectable.** Every component is open source with a traceable, publishable methodology.

**Novelty framing (important for the team).** Smartphone facial capture at roughly one-millimeter accuracy is established prior art. The contribution is not "a phone can measure a face." It is the photorealistic radiance-field reconstruction, the integrated documentation pipeline, and longitudinal tracking. Keep claims there.

---

## 2. Design principles

1. **Metric by construction and by validation.** The reconstruction is produced at metric scale and independently checked against physical measurements.

2. **Faithful to the abnormal.** No healthy-anatomy prior anywhere in the pipeline.

3. **File-based staging.** Each heavy learned model is its own stage with its own environment, writing files. No stage imports another's environment. Each stage/environment should exist independently within the same repository with orchestration handled outside of the individual components.

4. **The provider does nothing but record.** Every technical decision (scale, masking, how much to trust any prior) is automatic. There are no clinician-facing knobs.

5. **Modular stages with frozen contracts.** Each stage's inputs and outputs are fixed file formats and coordinate conventions, so components can be swapped and/or removed.

6. The code here may implement open-source software, regardless of commercialization license. This project's sole purpose at this time is for research and publication.

---

## 3. Architecture overview

**Data flow (each arrow is a set of files on disk):**

```
[Stage 1: Capture app]  (iPhone, rear camera + rear depth sensor)
      | color frames, metric depth, depth confidence, intrinsics, per-frame metric camera pose, timestamps
      v
[Stage 2: Front end]    (Depth Anything 3, nested giant-metric model; optional Free Geometry refinement)
      | camera poses, dense metric point cloud, per-frame depth + confidence
      v
[Stage 3: Metric alignment / validation]  (custom, model-agnostic)
      | metric-locked poses + point cloud, recovered scale factor, residual error report
      v
[Stage 4: Surface-normal prior]  (StableNormal; OPTIONAL, keep-or-remove flag)
      | per-frame normal maps (+ optional per-pixel trust weight)
      v
[Stage 5: Reconstruction host]  (MILo, with ported depth + normal supervision, optional MCMC densification)
      | optimizes surface-aligned Gaussians and extracts a mesh in the loop
      v
[Stage 6: Outputs]
        - refined Gaussian splat  (photorealistic, view-dependent radiance field)
        - metric surface mesh
```

**Environment strategy.** Create one isolated environment per heavy stage (Stages 2, 4, 5, and the metric cross-check tools). An orchestration script (Section 6) runs each stage as a subprocess and passes data by file path. Do not attempt to install all of these into one environment; their dependencies conflict (notably, the reconstruction host in Stage 5 pins an older toolchain).

---

## 3.5 Repository directory structure

The project is a single version-controlled repository (a "mono-repository," meaning one repository that holds all stages together). Each stage lives in its own folder and carries its own isolated software environment. These environments are never merged, because the stages require conflicting software versions, most notably the reconstruction host in Stage 5, which pins an older toolchain. A top-level coordinator runs the stages in sequence and passes data between them only as files on disk.

**Directory layout.**

```
project-root/
├── README.md
├── orchestrate.py                      # top-level coordinator (Section 6)
├── config/
│   └── pipeline.yaml                   # enabled stages, environment names, paths, thresholds
├── io_contracts/                       # the frozen file-format definitions (Section 5)
│   ├── README.md
│   ├── capture_session.md              # Stage 1 output contract
│   ├── frontend_output.md              # Stage 2 output contract
│   ├── metric_output.md                # Stage 3 output contract
│   ├── normals_output.md               # Stage 4 output contract
│   └── reconstruction_output.md        # Stage 6 output contract
├── common/                             # shared, dependency-light utilities (see constraint below)
│   ├── pyproject.toml                  # declares minimal dependencies only
│   ├── colmap_io.py                    # read and write the COLMAP camera format
│   ├── conventions.py                  # coordinate-convention helpers and conversions
│   ├── file_layout.py                  # session-folder path helpers
│   └── orientation_selftest.py         # the render-a-known-object test (Section 5)
├── stages/
│   ├── stage1_capture/                 # the iPhone application (Swift and Xcode)
│   │   ├── README.md
│   │   ├── CAPTURE_SPEC.md             # capture-application specification
│   │   └── (Xcode project files)
│   ├── stage2_frontend/
│   │   ├── environment.yml             # its own environment, never merged
│   │   ├── run.py                      # entry point (standard signature below)
│   │   └── README.md
│   ├── stage3_metric/
│   │   ├── environment.yml
│   │   ├── run.py
│   │   ├── METRIC_CONTRACT.md          # metric-module interface contract
│   │   └── README.md
│   ├── stage4_normals/
│   │   ├── environment.yml
│   │   ├── run.py
│   │   └── README.md
│   └── stage5_reconstruction/
│       ├── environment.yml
│       ├── run.py
│       └── README.md
├── sessions/                           # working data, one subfolder per capture (excluded from version control)
│   └── <session_id>/
│       ├── capture/                    # Stage 1 output
│       ├── frontend/                   # Stage 2 output
│       ├── metric/                     # Stage 3 output
│       ├── normals/                    # Stage 4 output (if enabled)
│       └── output/                     # Stage 6 output
└── tests/
    └── (contract and orientation tests)
```

**The dependency-light rule for `common/`.** The shared `common/` folder holds only small pure-Python helpers that every stage needs, such as reading and writing the camera format and converting coordinate conventions. It is a hard rule that `common/` depends on nothing beyond the Python standard library and `numpy`. This is what allows every stage, despite their conflicting environments, to install and import it without a version clash. Install it into each stage environment as a lightweight local package (`pip install -e ./common`). Anything heavier than `numpy` does not belong in `common/`; it belongs inside the stage that needs it.

**Environment naming and how the coordinator runs a stage.** Each stage's `environment.yml` names its environment after the stage, for example `name: pipeline_stage2_frontend`. The coordinator never imports any stage's code. It launches each stage as a separate process inside that stage's environment. With the `conda` environment manager this is a single command per stage, of the form:

```
conda run -n pipeline_stage2_frontend python stages/stage2_frontend/run.py --session sessions/<session_id> --config config/pipeline.yaml
```

**Standard entry-point signature.** Every Python stage exposes `run.py` accepting the same two arguments, `--session <path to the session folder>` and `--config <path to pipeline.yaml>`. Each `run.py` reads its inputs from, and writes its outputs into, the session folder, following the file contracts. This uniformity is what lets the coordinator treat every stage identically and lets any stage be replaced without touching the others.

**What `config/pipeline.yaml` holds.** A list of stages with an enable-or-disable flag each (so Stage 4 can be turned off per the keep-or-remove decision), the environment name and entry point for each stage, the numeric thresholds used by Stage 3, and any path settings. The coordinator reads this to decide which stages to run and how to invoke them.

---

## 4. Stages

Each stage below lists: **Purpose**, **Chosen component** and why it is currently best, **Repository to review**, **Environment and packages**, **Inputs (file contract)**, **Outputs (file contract)**, **Integration notes**, and **Swap candidates**.

---

### Stage 1: Capture application

**Purpose.** Record synchronized color, metric depth, depth confidence, camera intrinsics, and a metric camera path, and write them to a per-session folder. This replaces the third-party Record3D app so the team owns the full capture stack and its data provenance.

**Chosen approach and why.** Use Apple's direct capture framework (`AVFoundation`) with the built-in **LiDAR Depth Camera** device, which fuses the rear Light Detection and Ranging (LiDAR) depth sensor with the rear color camera. Compared with the augmented-reality framework (`ARKit`), this path provides higher-resolution color (up to 12 megapixels) and higher-resolution depth (up to 768 by 576 for still capture, versus 256 by 192 from `ARKit`), and it allows disabling Apple's depth smoothing so the raw depth is received with low-confidence points excluded, which is preferable for measurement. In parallel, run a lightweight `ARKit` world-tracking session solely to log the metric camera path, since that path is one independent way to lock true scale downstream.

**Repository and references to review.**

- Apple sample, capturing depth with the LiDAR camera: https://developer.apple.com/documentation/avfoundation/additional_data_capture/capturing_depth_using_the_lidar_camera

- `AVFoundation` depth data: https://developer.apple.com/documentation/avfoundation/avdepthdata

- `ARKit` world tracking and camera transform (for the parallel pose log): https://developer.apple.com/documentation/arkit/arcamera

- Data-format reference (how a comparable capture app structures its output, and the format the reconstruction ecosystem already ingests): Spectacular AI examples, https://github.com/SpectacularAI/sdk-examples and the reference app Record3D, https://github.com/marek-simonik/record3d

**Environment and packages.** Swift and Xcode. Target recent iOS. No Python. Requires an iPhone Pro or iPad Pro with a rear LiDAR sensor.

**Inputs.** None (this is the source stage).

**Outputs (file contract), one folder per capture session:**

- `rgb/000001.png` ... high-resolution color frames (lossless `.png` preferred for fidelity).

- `depth/000001.npy` ... metric depth per frame, 32-bit floating point, units of meters, shape `[H, W]`.

- `confidence/000001.png` ... per-pixel depth validity mask per frame (255 for a valid reading, 0 for an invalid one), derived from which depth pixels returned a reliable measurement. Graded confidence (low, medium, high) is an optional upgrade, not required.

- `intrinsics.json` ... the 3-by-3 camera intrinsic matrix (focal lengths and principal point) and image size.

- `poses.json` ... per-frame metric camera pose (position and orientation) from the parallel tracking session.

- `timestamps.json` ... per-frame timestamps, used to align the color, depth, and pose streams.

- Fix the **computer-vision (OpenCV) coordinate convention** for everything written here (camera looks down positive z, x right, y down), and record this explicitly in a `README` in the folder.

**Integration notes.** Timestamp-align the depth, color, and pose streams offline if the two sessions run at different rates. If running both sessions simultaneously proves awkward on device, fall back to leaning on the depth-based and physical-marker scale anchors in Stage 3 and drop the pose anchor.

**Swap candidates.** Record3D or the Spectacular AI software development kit (SDK) can produce equivalent per-session data during early development, before the custom app exists. Their output must be converted to the file contract above.

---

### Stage 2: Front end (camera poses and dense geometry)

**Purpose.** From the color frames, recover the camera pose for every frame and a dense point cloud, which is the geometric scaffold for everything downstream.

**Chosen component and why it is best (verified July 2026).** **Depth Anything 3**, using the nested giant-plus-metric model `DA3NESTED-GIANT-LARGE`. This is a feed-forward model, meaning it produces its result in a single pass rather than by slow iterative optimization. As of mid-2026 it is the most accurate feed-forward geometry model available, reported to surpass the prior leader by roughly 44 percent on camera pose accuracy and 25 percent on geometric accuracy. Two properties matter for this project: the nested model outputs geometry **already in metric units (meters)**, which supports the metric goal directly, and the model can be conditioned on known camera poses if desired.

**Optional accuracy add-on.** **Free Geometry** (posted April 15, 2026) is a test-time refinement that lets a frozen model like Depth Anything 3 quietly retune itself to a specific scan with no ground truth, in under two minutes, for a reported three-to-four percent accuracy gain. For a publication pipeline where two minutes of extra compute is negligible, enable this.

**Repositories to review.**

- Depth Anything 3: https://github.com/ByteDance-Seed/Depth-Anything-3

- Model card for the nested metric model: https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE

- Free Geometry (optional): https://github.com/hiteacherIamhumble/Free-Geometry

**Environment and packages (isolated environment).**

- Python 3.10 or newer.

- Install: `pip install xformers "torch>=2" torchvision`, then from the cloned repository `pip install -e .`.

- The Gaussian output head additionally requires the splatting library at a pinned commit: `pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70`.

- For the optional interactive viewer: `pip install -e ".[app]"`.

- Model weights download from the Hugging Face hub on first use (`depth-anything/DA3NESTED-GIANT-LARGE`).

**Inputs (from Stage 1).** `rgb/*.png`. Optionally `intrinsics.json` and `poses.json` if pose-conditioning is used.

**Outputs (file contract).**

- Per the model application programming interface (API), the model returns: `depth` `[N, H, W]` float32 in meters, `conf` `[N, H, W]` float32, `extrinsics` (world-to-camera) `[N, 3, 4]`, `intrinsics` `[N, 3, 3]`.

- Write these to disk as: `frontend/poses/` (the extrinsics and intrinsics), `frontend/depth/*.npy`, `frontend/conf/*.npy`, and a fused dense point cloud `frontend/points.ply`.

- Also emit the same camera data converted to **COLMAP sparse-model format** (`cameras.bin`, `images.bin`, `points3D.bin`) under `frontend/colmap/sparse/0/`, because the Stage 5 reconstruction host reads that format. See Section 5 for the conversion.

**Integration notes.** The model API can export directly to several formats via `export_format` (`glb`, `npz`, `ply`, `gs_ply`, `gs_video`); use `ply` and `npz` for the point cloud and arrays. Prefer the `use_ray_pose` option for slightly more accurate poses if runtime permits.

**Swap candidates.** MapAnything (`facebook/map-anything`), which is metric by design and can also wrap Depth Anything 3 internally, or AMB3R (a metric feed-forward model highlighted at the 2026 computer-vision conference; locate its official repository before use). These also serve as the Stage 3 cross-checks.

---

### Stage 3: Metric alignment and validation (custom, model-agnostic)

**Purpose.** Guarantee true metric scale, and produce a residual error number that becomes the pipeline's headline accuracy metric. Even though Stage 2's nested model outputs metric geometry, that scale is a learned estimate, so it is cross-checked and, if needed, corrected against physical measurements.

**Concept.** A reconstruction can be off by a single overall size factor (and offset). Solve for that factor by aligning the reconstruction to real-world measurements, then apply it. Three independent physical anchors are available, and agreement among them is the evidence of metric validity:

1. The depth sensor's metric readings (Stage 1 `depth/`), on high-confidence pixels only.

2. The metric camera path (Stage 1 `poses.json`), matched against the front end's estimated camera positions. This anchor does not depend on the noisy depth.

3. An optional physical object of known size in frame (a sterile ruler or printed marker), the only true physical ground truth of the three.

**Chosen approach.** Build this as a standalone, model-agnostic module. Estimate the similarity transform (scale, rotation, translation) that best aligns the reconstruction to each anchor using a closed-form least-squares method (the Umeyama algorithm), refined by Iterative Closest Point. Compare the scale from the nested model against the scale implied by each physical anchor. If they agree, report the residual. If they disagree beyond a threshold, prefer the physical anchors and **flag** the session rather than silently averaging.

**Repositories to review (for the cross-check models and the alignment primitives).**

- MapAnything (independent metric cross-check): https://github.com/facebookresearch/map-anything

- Open3D (provides Iterative Closest Point and point-cloud tooling): https://github.com/isl-org/Open3D

**Environment and packages.** Python. `numpy`, `scipy`, `open3d`. Optionally the MapAnything environment (Python 3.12, `pip install -e ".[all]"`, model `facebook/map-anything`) for the independent metric cross-check.

**Inputs.** Stage 2 `frontend/` outputs, plus Stage 1 `depth/`, `confidence/`, and `poses.json`.

**Outputs (file contract).**

- `metric/points_metric.ply` ... the metric-locked dense point cloud.

- `metric/colmap/sparse/0/` ... metric-locked camera model in COLMAP format for Stage 5.

- `metric/scale_report.json` ... the recovered scale factor, the per-anchor scale estimates, the residual error, and a pass or flag status.

**Integration notes.** Keep this module independent of which front end produced the reconstruction, so it works unchanged if Stage 2 is swapped. The `scale_report.json` residual is what gets reported in the paper and compared against the Canfield Vectra reference in validation.

**Swap candidates.** Not applicable; this is bespoke glue that should remain.

---

### Stage 4: Surface-normal prior (OPTIONAL, keep-or-remove flag)

**Purpose.** Provide a surface-normal map per frame, meaning the direction each patch of surface faces, to gently regularize the reconstruction in smooth, textureless regions where the image alone is ambiguous and the depth sensor is jittery. This stage is a **flag**: it can be kept or removed without damaging the pipeline, and whether it earns its place is settled empirically by the experiment in Section 8.

**Why this is a flag, not a fixture.** A learned normal model helps most on smooth skin, where the sensor's depth is noisy (jittery) but not wrong, and clean predicted normals suppress that jitter. On wounds and other wet, dark, or textureless surfaces, all learned normal models are unreliable, so the prior must never dominate. It is therefore added at a modest fixed weight, kept small relative to the two trusted signals (the actual image matching and the metric depth), and combined with the existing depth-confidence masking so it steps back where the sensor reports low confidence. Because its weight is modest and it is confidence-gated, its downside is bounded: it can only help or do little, not actively distort.

**Chosen component and why.** **StableNormal**. It is the general-purpose normal estimator specifically demonstrated to regularize Gaussian-splatting surface reconstruction and to yield the lowest surface error on standard benchmarks, and it is comparatively robust under difficult lighting and low-quality imaging. It is a **general-scene geometry model**, not a human-anatomy model, so it carries no "healthy anatomy" assumption, which is why it is acceptable where the excluded human-specific model (Section 12) is not.

**Repository to review.** https://github.com/Stable-X/StableNormal

**Environment and packages (isolated environment).** Python. It is a diffusion-based model; install `torch`, `torchvision`, `diffusers`, `xformers`, `numpy`, `Pillow`. It can be loaded via `torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", ...)` or from the cloned repository; weights cache to the local `torch` hub directory on first use.

**Inputs (from Stage 1 and Stage 3).** `rgb/*.png` for the normal prediction; Stage 3 metric depth is used to compute the optional per-pixel trust weight.

**Outputs (file contract).**

- `normals/000001.npy` ... per-frame normal map, shape `[H, W, 3]`, values in the range negative one to one, expressed in the **camera frame in the computer-vision (OpenCV) convention**, matching the layout the reconstruction host expects for externally supplied normals.

- Optional `normals_weight/000001.npy` ... per-pixel trust weight in the range zero to one. Start without this (fixed uniform weight); add it only if the Section 8 experiment shows the prior distorting deep wounds, in which case tie the weight to the Stage 1 confidence map.

**Integration notes.** Everything here is automatic; there is no clinician input. Match the normal-map file layout and coordinate convention to what Stage 5's supervision term reads, and verify orientation with the render-a-known-object test in Section 5.

**Swap candidates.** DSINE (https://github.com/baegwangbin/DSINE), a lighter feed-forward normal estimator already integrated in the reconstruction ecosystem; or compute normals directly from the Stage 3 metric depth with no learned model at all (the most bias-free option, at the cost of noisier normals). The Section 8 experiment compares exactly these options.

---

### Stage 5: Reconstruction host (Gaussian splatting with mesh in the loop)

**Purpose.** Build the actual reconstruction: optimize a set of Gaussian splats to reproduce the captured images while respecting the metric depth and the normal prior, and extract a surface mesh during optimization rather than as a lossy afterthought.

**Chosen component and why it is best (verified July 2026).** **MILo (Mesh-In-the-Loop Gaussian Splatting)**. It differentiably extracts a mesh from the Gaussians at every optimization step, with information flowing both directions, so the splats are pushed to sit cleanly on the true surface and the mesh stays consistent with them. It reaches state-of-the-art surface quality while producing meshes with an order of magnitude fewer vertices than prior methods, and as of mid-2026 nothing has clearly surpassed it for geometry-focused surface extraction. It already renders differentiable depth and normal maps and already integrates a monocular depth model for an optional regularization term, so adding the supervision this project needs fits its design.

**What the team must build into it.** MILo does not, out of the box, supervise against a real metric depth sensor or an external normal prior. Port in the following (these are loss-term concepts, since the source of the recipe lives in a different framework, see Swap candidates):

1. **Metric depth supervision.** An edge-aware depth loss driven by the Stage 3 metric depth, which trusts the sensor on smooth regions and less at edges. This is the loss recipe from the AGS-Mesh method.

2. **Normal supervision (if Stage 4 is enabled).** A cosine loss between the rendered normals and the Stage 4 normal maps, weighted by the fixed modest weight (and the optional trust weight).

3. **Confidence masking.** Discard unreliable depth using both the Stage 1 confidence map and the consistency masking from AGS-Mesh.

4. **Metric initialization.** Initialize the splats from the Stage 3 metric dense point cloud (back-projected sensor depth), rather than from sparse feature points. This improves reconstruction and is the root fix for "floaters" (stray blobs of geometry in empty space), which come from poor initialization.

**Optional densification upgrade.** MILo's built-in densification (the Mini-Splatting2 strategy) is strong. For extra robustness to initialization, optionally port the Markov Chain Monte Carlo (MCMC) densification, whose reference implementation lives in the same code lineage as MILo. The splatting library `gsplat` also implements this as `MCMCStrategy`, but that lives in a different framework, so for MILo prefer the same-lineage implementation.

**Repositories to review.**

- MILo: https://github.com/Anttwo/MILo

- AGS-Mesh and DN-Splatter (the depth-and-normal supervision recipe and confidence masking to port): https://github.com/maturk/dn-splatter

- Markov Chain Monte Carlo densification, same lineage as MILo (optional): https://github.com/ubc-vision/3dgs-mcmc

- Splatting library, reference `MCMCStrategy` and rasterization (optional reference): https://github.com/nerfstudio-project/gsplat

**Environment and packages (isolated environment, older toolchain).**

- Python 3.9, with a Compute Unified Device Architecture (CUDA) toolkit version of 11.8, and PyTorch 2.3.1, following the MILo repository, which has been tested on that configuration.

- MILo's rasterizer submodules and mesh tooling (the differentiable rasterizer variants, a nearest-neighbor helper, and `nvdiffrast` for differentiable triangle rasterization) build during install; follow the repository's setup exactly.

- COLMAP is required for data preparation and format tooling: https://github.com/colmap/colmap (install via the system package manager or from source).

**Inputs (from Stage 3 and Stage 4).**

- `metric/colmap/sparse/0/` ... metric camera model.

- `rgb/*.png` ... the color frames.

- `metric/points_metric.ply` ... initialization point cloud.

- Stage 1 `depth/` and `confidence/` ... for the depth supervision and masking.

- `normals/` (and optional `normals_weight/`) ... if Stage 4 is enabled.

**Outputs.** See Stage 6.

**Integration notes.** MILo consumes a COLMAP-format dataset, which is why Stages 2 and 3 emit that format. The MILo repository lists feed-forward pose ingestion as forthcoming; check for that before writing the COLMAP conversion, in case direct ingestion of Stage 2 poses becomes available. MILo's functional interface (mesh sampling, signed-distance initialization, mesh extraction) is designed to plug into any project following the original Gaussian-splatting template, which is the seam through which the ported supervision attaches.

**Swap candidates.** The DN-Splatter and AGS-Mesh host itself, which already contains the depth and normal supervision and confidence masking as first-class features and runs on the `gsplat` library, at the cost of a post-hoc rather than in-the-loop mesh. This is the natural fallback if porting the supervision into MILo proves heavy.

---

### Stage 6: Outputs

**Purpose.** Emit both deliverables, which MILo produces from a single optimization.

**Outputs (file contract).**

- `output/point_cloud.ply` ... the refined Gaussian splat: the optimized, surface-aligned Gaussians. This is the photorealistic, view-dependent radiance field, the record of how the tissue actually looks from different angles. This is the appearance and archival side of the invention.

- `output/mesh.ply` (and optionally `.obj`) ... the metric surface mesh extracted in the loop. This is the measurement side.

- `output/renders/` ... preview renders for quality inspection.

- `output/provenance.json` ... a record of component versions, model identifiers, the Stage 3 scale report, and capture metadata, for reproducibility and the eventual regulatory pathway.

---

## 5. The intermediate data contract (formats and conventions)

Freeze these before parallel work begins, so the capture, front-end, normal, and reconstruction efforts can proceed against fixed boundaries.

**Coordinate convention.** Use the computer-vision (OpenCV) convention **everywhere** (camera looks down positive z, x right, y down; camera poses stored as world-to-camera unless a stage's format dictates otherwise). Convert exactly once at each boundary. Mismatched conventions, especially for normals, are the classic silent failure of this kind of pipeline.

**Units.** All depth and geometry in meters, stored as 32-bit floating point.

**Per-frame arrays.** Store as `.npy` for lossless numeric fidelity (depth, confidence, normals), and color as lossless `.png`.

**Camera model for the reconstruction host.** COLMAP sparse-model format (`cameras.bin`, `images.bin`, `points3D.bin`). The Stage 2 model API returns world-to-camera extrinsics `[N, 3, 4]` and intrinsics `[N, 3, 3]`; convert these into COLMAP `images.bin` (one entry per frame, rotation as a quaternion plus translation) and `cameras.bin` (a pinhole model with the intrinsics), and populate `points3D.bin` from the dense cloud or leave sparse if the host only needs poses and an initialization point cloud supplied separately.

**Normal-map layout.** `[H, W, 3]`, values in negative one to one, camera frame, computer-vision convention, one file per frame under `normals/`, matching the layout the reconstruction host's supervision term expects.

**Orientation self-test (mandatory).** Before trusting any result, add one automated test that renders a known synthetic object and confirms its surface normals point outward and its depth increases away from the camera. Run it whenever a convention at any boundary changes.

---

## 6. Orchestration

Write a top-level orchestration script (Python is fine) that:

1. Takes a capture-session folder as input.

2. Activates each stage's environment and runs it as a subprocess, passing data by file path only.

3. Writes each stage's outputs into a session-scoped working directory following the contracts above.

4. Halts and reports if Stage 3 raises a scale-disagreement flag.

5. Records component versions and model identifiers into `provenance.json`.

Because stages communicate only through files, the orchestration script never imports any model code directly, which is what keeps the conflicting environments isolated. A simple approach is one shell entry point per stage plus a coordinating script; a workflow manager is optional and not required for a first version.

---

## 7. Build order (to prove the risky parts first)

1. **Stage 3 core plus the metric experiment (Section 8, Experiment A), on existing captures.** Validates the entire metric idea in week one, before any new software exists.

2. **The normal-prior experiment (Section 8, Experiment B), on existing captures.** Settles empirically whether Stage 4 belongs at all, before anyone builds it in.

3. **Stage 5 reconstruction upgrades on existing data:** metric initialization, the ported depth loss, the confidence masking, then (if Experiment B supports it) the normal loss, then optionally the MCMC densification. These are the largest quality gains and need no new capture.

4. **Stage 2 front end** wired to real data and to the COLMAP conversion.

5. **Stage 1 capture app**, once the pipeline is proven worth feeding better data into.

6. **Full validation** against the Canfield Vectra reference (Section 9).

---

## 8. Two week-one experiments (run on existing scans before building)

**Experiment A: does a single global scale make the reconstruction metric?**

Run Stage 2 on ten existing scans. Using the physical anchors from Stage 3, fit one global scale factor (and offset) against the sensor depth, apply it, and measure whether the residual error is uniform across frames or drifts frame to frame. Uniform residual confirms the easy single-scale case and validates the whole metric approach. Record the residual per scan.

**Experiment B: does the learned normal prior actually help, and does it ever hurt?**

On the same scans, reconstruct under three conditions and measure surface accuracy against your reference:

1. no normal prior at all,

2. normals computed directly from the sensor depth (no learned model),

3. StableNormal at a fixed modest weight.

Read off the result: if condition 3 clearly improves smooth regions without distorting wounds, keep Stage 4 as specified with no gating. If it distorts deep wounds (where the sensor goes blank and a wrong flat prior could make a deep wound look shallower, which is clinically unacceptable), add the single confidence-tied weight described in Stage 4. If it does not clearly help in any condition, remove Stage 4 entirely. No component earns a place by assertion; it earns it by measurably improving reconstructions.

---

## 9. Validation protocol

- **Accuracy target.** Match or beat roughly one millimeter of surface deviation against a gold-standard reference. Report the Stage 3 scale factor and residual as the quantitative accuracy number.

- **Reference comparison.** Benchmark the full pipeline against the Canfield Vectra system and against the collaborator's photogrammetry pipeline on the porcine study.

- **Metric triangulation.** Report agreement across the three Stage 3 anchors as evidence of metric validity.

- **Wound characterization.** Document, with data, where the sensor depth alone suffices and where it struggles. That boundary is both a limitation to disclose and a publishable finding.

---

## 10. Open flags and decisions to track (edit this section over time)

- **Stage 4 (normal prior): keep or remove.** Decided by Experiment B. Currently included as a bounded, automatic, modest-weight regularizer. Removable without damaging the pipeline.

- **Stage 4 trust weighting: fixed versus confidence-tied.** Start fixed; add confidence-tied weighting only if Experiment B shows wound distortion.

- **Stage 5 densification: built-in versus MCMC.** Start with MILo's built-in strategy; add same-lineage MCMC only if initialization robustness needs it.

- **Stage 5 host: MILo versus DN-Splatter or AGS-Mesh.** MILo for best in-the-loop mesh; the DN-Splatter or AGS-Mesh host is the fallback if porting supervision proves heavy.

- **Front end: nested metric model versus relative model plus custom scale.** Currently the nested metric model, cross-checked by Stage 3. Revisit if a more accurate model appears.

- **Licensing.** Deferred. Several components carry non-commercial research licenses, acceptable for publication. Revisit before any commercial use.

---

## 11. Repository index

| Stage | Component | Repository |
| --- | --- | --- |
| 1 | Apple depth capture sample | https://developer.apple.com/documentation/avfoundation/additional_data_capture/capturing_depth_using_the_lidar_camera |
| 1 | Capture-format reference (Spectacular AI) | https://github.com/SpectacularAI/sdk-examples |
| 1 | Capture-format reference (Record3D) | https://github.com/marek-simonik/record3d |
| 2 | Depth Anything 3 | https://github.com/ByteDance-Seed/Depth-Anything-3 |
| 2 | Nested metric model card | https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE |
| 2 | Free Geometry (optional refinement) | https://github.com/hiteacherIamhumble/Free-Geometry |
| 3 | MapAnything (metric cross-check) | https://github.com/facebookresearch/map-anything |
| 3 | Open3D (alignment primitives) | https://github.com/isl-org/Open3D |
| 4 | StableNormal (optional prior) | https://github.com/Stable-X/StableNormal |
| 4 | DSINE (normal-prior alternative) | https://github.com/baegwangbin/DSINE |
| 5 | MILo (reconstruction host) | https://github.com/Anttwo/MILo |
| 5 | DN-Splatter and AGS-Mesh (supervision recipe; fallback host) | https://github.com/maturk/dn-splatter |
| 5 | 3D Gaussian Splatting as Markov Chain Monte Carlo (optional densification) | https://github.com/ubc-vision/3dgs-mcmc |
| 5 | gsplat (splatting library, reference) | https://github.com/nerfstudio-project/gsplat |
| 5 | COLMAP (data prep and format) | https://github.com/colmap/colmap |

---

## 12. Considered and deliberately excluded

- **Sapiens2 (human-specific surface model).** Excluded from the reconstruction path. It is trained on healthy human anatomy and carries an internal expectation of what a human body and face should look like. For a tool whose purpose is to faithfully document wounds, deformities, and post-surgical anatomy, that healthy-anatomy expectation is a liability, because it could regularize an anomaly toward normal in a way that is clinically misleading. Its only advantage was sharper detail on healthy skin, which is not worth that risk. The general-scene normal model in Stage 4 (StableNormal) is used instead precisely because it carries no anatomical assumption. Sapiens2 may still be relevant to a separate, future project (a wound and anomaly geometry model), which is out of scope here.

---

*End of specification. Edit stage by stage as decisions are made.*

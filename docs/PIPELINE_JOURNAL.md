# PIPELINE JOURNAL — comprehensive working record & autonomous-operation handbook

**Written 2026-07-07 ~19:30 by the working agent, mid-marathon, at the owner's direction.**
Purpose: this is NOT a summary — it is the full working state of the investigation into making this
pipeline strong and faithful, written so that a future agent (or the same agent after disconnection)
can resume seamlessly, and so every finding carries its evidence and its "why." Read this first;
then `docs/SWEEP_RESULTS.md` (A/B tables), `docs/DAILY_NOTES.md` (day narratives),
`docs/EXPERIMENTS_BACKLOG.md` (audit + matrix), `docs/MILO_PLAN.md` (MILo build/gotchas).

---

## 0. How to use this document (disconnection recovery)

1. **Check the overnight runner first:** `sessions/_sweep_eval/overnight/status.log` (live) and
   `MORNING_SUMMARY.md` (written when all arms finish). Runner script:
   `scripts/overnight_matrix.sh`, launched detached (nohup+disown, lockfile `overnight/runner.lock`
   holds its pid). It is IDEMPOTENT — safe to re-run anytime; it skips finished arms, self-heals
   half-finished ones, waits for any active MILo process, runs strictly serially.
2. `pgrep -af "envs/milo/bin/python"` tells you what's training. ONE quality run owns the GPU at a
   time (§7 VRAM discipline — this is a hard rule, learned by OOM).
3. Every experiment's numbers live in `sessions/_sweep_eval/<name>/stats.json` + renders
   (`comparison.png`, `recon_*.png`, `face_tight_*.png`). Reconstructions live in
   `sessions/<session>/output_<label>/{point_cloud.ply, mesh.ply, provenance_stage5.json}`.
4. The task list with decision rules is §10. The morning analysis protocol is §11.

---

## 1. Mission

Clinical pipeline: short iPhone captures of patient anatomy → **metric AND photorealistic** 3D
Gaussian splat + **mesh** for health records. Target ~1 mm surface deviation vs a Canfield Vectra
reference. Principle: "faithful to the abnormal" — no healthy-anatomy priors. The current campaign
(owner-directed): resolve the **granularity/fidelity** deficit in outputs, decide the **capture
method** (ARKit vs HQ-Depth), the **role of LiDAR** (surface supervision vs scale anchor), and
**DA3's place** — then declare the strongest pipeline configuration.

## 2. The two test subjects (know your data)

- **Feet captures (the clinical case), 2026-07-04, same feet, 74 s apart:**
  - `sessions/session_20260704_143210` — ARKit capture: 57 frames 1920×1440 video-grade + VIO poses
    + 256×192 LiDAR. Metric colmap = DA3 pose-conditioned, **dense 100k-point baked init**.
  - `sessions/session_20260704_143324` — HQ capture: 47 frames 4032×3024 stills + 320×240 raw LiDAR,
    NO poses. Metric colmap = **from-scratch SfM** (47/47 @ 1.50 px) + **LiDAR metric-lock**
    (`scripts/pose_ba/03b_relock_lidar.py`: S=0.078135, MAD 1.0% over 32,048 ray comparisons)
    → **sparse 17k-point init**. DA3 artifacts preserved in `metric_da3/`, `output_da3_nomesh/`.
  - Both are SCENE-scale (feet small in frame, stand + floor dominate) — a known input weakness.
- **Face captures (the owner-reference subject), Record3D export, 3,258 JPGs 1440×1920:**
  `sessions/Previous face photos`. Owner's old preliminary run (COLMAP→MILo, no depth, no dense,
  ~169 frames, old T4 machine) looked visibly clean = the quality bar. Face fills the frame (high
  pixel coverage). No depth data → face runs are gauge-free (non-metric) by design.
  Test session: `sessions/face_depthfree_test/` (v1/v2/v3 arms, `scripts/face_depthfree_test.py`).

## 3. ROOT CAUSES FOUND (each cost real time; do not rediscover)

1. **nvdiffrast 2048-px cap = the historic "HQ MILo crash".** nvdiffrast's CUDA rasterizer hard-caps
   at 2048 px/side; HQ rendered the in-loop mesh at 4032×3024 → CUDA 700 (illegal address,
   `fineRasterKernel`) at the FIRST mesh build (iter 8001 on fast). Pose- and density-independent;
   ARKit's 1920 was under the cap. v0.3.3's claimed >2048 auto-tiling fails on this build.
   **Fix (permanent):** `milo_supervised.py` caps `-r` so max(w,h) ≤ 2048 when mesh reg is on
   (HQ → -r2/2016). **NOT poses** (a coherent-SfM run crashed identically — hypothesis falsified by
   test). Fast repro trick: MILo checkpoints at 8000; `train.py --start_checkpoint chkpnt8000.pth`
   rebuilds the crashing mesh in ~30 s.
2. **The invisible "fast" schedule.** MILo's upstream default `--config_path ./configs/fast`
   (train.py:577) = 18k iters, **densify stops at iter 3,000**, importance-prunes 3k/8k. EVERY run
   in this repo ever used it unknowingly → every result was capacity-starved; nothing recorded the
   schedule. **Fix:** provenance now stamps `milo_schedule` + `mesh_config`; new schedules
   `configs/quality` (stock 30k/densify-15k) and `configs/quality_mid` (22k/densify-8k — the
   A6000-feasible point, see §7).
3. **`regularization_from_iter` MUST equal `densify_until_iter`.** The renderer that supplies
   `area_max` (needed by MS2 densification, train.py:378) switches to the regularization renderer at
   `regularization_from_iter` (default 3,000). fast "worked" only because both were 3,000. A longer
   densify window without moving reg_from → KeyError 'area_max' at iter 3,000. Both quality configs
   carry the aligned value.
4. **`mesh_extract_sdf.py --iteration` defaults to a hardcoded 18000.** A 30k run trains fine, then
   extraction fails (FileNotFoundError iteration_18000). **Fix:** driver now globs the saved
   `iteration_*` dir and passes `--iteration <latest>`.
5. **`radius_culling` (MILo's built-in "foreground culling") is a TRAP for standoff captures:** it
   keeps gaussians within the **camera-hull** (center = mean camera position, radius = camera
   spread). Feet shot from ~0.85 m sit OUTSIDE the hull → it would cull the subject. Only valid for
   orbits AROUND a subject. Do not use for feet-style captures.
6. **nvdiffrast ~16.7M-triangle (2^24) limit = the R2' qualmid crash (found 2026-07-08).** Second,
   DISTINCT nvdiffrast CUDA-700: `triangleSetupKernel`, at quality_mid's first mesh phase on the HQ
   feet. Choke-point instrumentation (unconditional stats print inside `nvdiff_rasterization`)
   proved the geometry PERFECTLY CLEAN (finite, in-range, 2016×1512) but the in-loop mesh reached
   6-8.5M verts / 4.5-12M+ faces and spiking past 2^24 — the documented limit MILo's own
   `ScalableMeshRenderer` exists to chunk. **Fix: `use_scalable_renderer: true`** in
   `configs/mesh/quality{,_mid}.yaml` (config switch, zero code). Explains why ARKit (≤10.7M faces)
   and face (3.7M) passed. **Chapter 2 (same day):** scalable-at-2^24 STILL failed — nvdiffrast's
   "subtriangle count overflow" (torch_rasterize.cpp:123) fires on triangle DENSITY within one ≤2^24
   chunk (a 12M-face mesh = 1 pass, chunking never engaged). **Final fix:** ScalableMeshRenderer
   `max_triangles_in_batch` default 2^24 → **2^22** (scene/mesh.py, clinical-pipeline patch) →
   real 3-pass chunking; VALIDATED from chkpnt12000: 1,700+ iters / 6,900+ rasterize calls past the
   crash point, losses sane (fuse_fragments gradients correct). Mesh-phase cost ≈1.1 s/it at 12M
   faces. Extraction note: pass a scalable-enabled --config to mesh_extract_sdf for big meshes. Diagnosis pattern that worked: instrument the CHOKE POINT all paths
   funnel through (scene/mesh.py:nvdiff_rasterization), print to STDERR, throttle+always-on-bad;
   NOTE the tqdm \r-gluing trap — extract logs with `tr '\r' '\n'` BEFORE grepping, and never
   filter display greps with `-v "it/s"` (it silently dropped glued diagnostic lines).
7. **Historic trimesh + EGL + scale gotchas** (earlier sessions): system python needed pip bootstrap
   for trimesh; nvdiffrast must use RasterizeCudaContext headless; INRIA rasterizer overflows on
   ~0.1-unit metric scenes → driver auto-scales S=1/nerf_radius and inverts on output (see
   MILO_PLAN build log).

## 4. THE LEVERS — effect sizes with evidence (all numbers = mean dihedral°, subject-cropped where noted)

| Lever | Effect | Evidence (dirs under the session) | Status |
|---|---|---|---|
| **depth_lambda (LiDAR surface supervision)** | THE bumpiness driver. λ0.2→0: ARKit 21.1→15.8, HQ 18.3→14.6 (subj 18.9→14.8). **At capacity it AMPLIFIES**: qualmid λ0.2 = 32.7 subj (see R1''). Mechanism: supervising against noisy 256×192/320×240 sensor stamps noise; more capacity fits noise finer. COST of λ0: floater control (HQ λ0 cloud z-extent 2.6→6.6 m — LiDAR was a free-space suppressor). | `output_arkit_dense` vs `output_arkit_depth0`; `output_hq_dense` vs `output_hq_depth0`; `output_arkit_R1_qualitymid` | fast-matrix proven; capacity λ-pair = tonight's R4'/R3'; λ0.05 whisper = R6a |
| **Training schedule** | fast(densify→3k) vs stock(→15k): face +27% head verts, real stubble micro-relief, +3° dihedral (texture), more floaters. Final COUNT barely moves (distillation, see §5.3): 57.7k→80.5k. On feet nondense: budget 46.9k→441k (9.4×!) because feet-fast was init/window-starved. | `face_depthfree_test/output` vs `output_quality`; R1'' | measured; quality_mid is the feet standard going forward |
| **Image-set selection (count + sharpness)** | v1 blind-172 vs v3 sharpest-per-9 (362): visibly MORE COMPLETE face (mouth/chin holes filled), +18% verts, SAME roughness, tighter subject cluster. THE input lever for completeness. | `face_depthfree_test/output` vs `output_v3`; `_sweep_eval/face_imageset/face_tight_v1v3.png` | proven; capture protocol implication §12 |
| **dense_gaussians** | fast: dense→smoother-OFF? No: dense 406k/21.1° vs non-dense 46.9k/18.4° (dense = rougher, more detail). At quality schedule: VRAM-infeasible (>55 GB proj). MS2 distillation makes its final-count benefit moot. | `output_arkit_dense` vs `output_arkit_lowdens` | dropped from matrix; revisit only with capped window + isolation |
| **mesh_config (tet-grid res)** | lowres = ROUGHER (41.4 vs 21.1) — coarser sampling of same surface, NOT smoothing. highres untested (R6 candidate after isolation). | `output_arkit_meshlowres` | dead end for smoothness |
| **Simplification retention percentile** | UNTESTED. `init_cdf_mask` keeps top-99% importance mass → final counts collapse (16.4M→80k face). If more FINAL gaussians are wanted, this percentile (in `gaussian_model.py`) is the knob, not schedule length. | — | R6 candidate |
| **Subject isolation** | UNTESTED (design ready). Photometric-loss image-mask (gradient source!) + box-prune + output-crop; subject box via camera-optical-axis intersection (background-independent). Init-crop alone CANNOT work (densification is gradient-driven; verified in code). `radius_culling` unsuitable (§3.5). | design in EXPERIMENTS_BACKLOG "Implementation notes" | top structural lever for feet; post-matrix |
| **Capture practice (fill-frame, video-dense)** | The dominating input lever. Face (fills frame, 172-362 views) = clean; feet (small subject, 47-57 sparse frames) = poor under every knob. No reconstruction knob adds pixels. | face vs feet, everything | owner protocol change; §12 |
| **Poses** | NOT a lever. Three methods agree 1-2 mm; from-scratch SfM registers 100% (feet 47/47 @1.5 px; face 172/172 @0.79, 362/362 @0.72; focal solves agree <1 px across independent runs). | pose_ba experiments; face SfM gates | settled — SfM is reliable; VIO convenient |
| **DA3** | Not required for reconstruction (COLMAP sparse init suffices: face runs). Its dense init DID mask the fast-schedule starvation on ARKit feet (100k baked pts vs SfM's 17k — init×window interaction §5.4). Remaining value: pose-conditioning convenience + fallback for SfM-hostile captures. | face runs; ARKit-vs-HQ @ fast | demote to optional/fallback pending matrix |

## 5. MECHANISMS (the theory that the data supports — use these to predict)

1. **Noise-stamping:** per-pixel depth supervision against a low-res noisy LiDAR imprints sensor
   noise on the surface. Direction: proven twice (both pose paths). Capacity-dependence: proven
   (R1'': 2× capacity nearly 2× roughness at λ0.2, while depth-free face barely changed).
2. **Metric-through-poses:** with the COLMAP model metric-locked (VIO or SfM+LiDAR-lock), global
   scale is inherited structurally — cameras are FIXED — so λ=0 does NOT lose metric scale.
   Verified: subject-cluster dims agree ≤~2% across λ arms. Residual box differences = floaters.
   → **LiDAR's metric job moves to the anchor (03b) / VIO; its surface job should end.**
3. **Distillation invariance:** MS2 importance-pruning (simp2) collapses final counts to the same
   magnitude regardless of peak (16.4M→80k face; feet qualmid kept 441k — scene complexity
   dependent). Schedule buys better *placement*, not more *count*. Retention percentile = the count knob.
4. **Init-density × densify-window interaction:** sparse SfM init (17k) + short window (3k) = the
   HQ-fast disaster (diffuse blob); dense DA3 init (100k) partially masked it on ARKit-fast. A long
   window lets a sparse init catch up (R2'/R4' test this tonight).
5. **Pixel-coverage ceiling:** detail cannot exceed what capture pixels sampled. Fill-frame face ≫
   standoff feet under every configuration.
6. **LiDAR as incidental floater suppressor:** the depth term also penalized free-space gaussians;
   λ0 needs a replacement (isolation, box-prune, or λ≈0.05 = R6a).

## 6. CODE MAP — what we changed and where the bodies are buried

- `stages/stage5_reconstruction/milo_supervised.py` — THE driver. Our changes: nvdiffrast `-r` cap;
  `mesh_config` passthrough; `milo_schedule` passthrough (`--config_path configs/<name>`); extraction
  `--iteration` from saved dir; provenance stamps schedule+mesh_config. Pre-existing: S=1/nerf_radius
  scene scaling (+ LiDAR scale threading), INRIA-ply scale-down, `_object_box` mesh crop.
- `third_party/MILo/milo/configs/{fast,quality,quality_mid}` + `configs/mesh/{default,quality,quality_mid}.yaml`
  — schedule files (config **overrides** CLI: arguments/__init__.py:139). quality_mid = 22k iters,
  densify→8k, simp 8k/12k, reg_from 8k, mesh 12k→22k, ckpts [8k,12k].
- `third_party/MILo/milo/train.py` — flag-guarded LiDAR loss (H2 port; `lidar_kick_on = λ>0 AND dir`,
  line 188 — λ=0 → whole block off, so depth-free needs NO code change). We did NOT otherwise modify
  MILo (owner directive: no bandages; instrumentation was added then fully reverted).
- `scripts/face_depthfree_test.py` — face arms: --variant {v1,v3} × --schedule {fast,quality,quality_mid};
  sharpness selection (cv2 Laplacian per 9-frame window); pycolmap SfM per the owner's old recipe
  (PINHOLE, BA-refined focal — init fx 1367 = device-K scaled, refined to ~1496).
- `scripts/pose_ba/03b_relock_lidar.py` — HQ metric lock: per-observation LiDAR/SfM-depth ratio
  median (gauge-immune). `03_relock.py` is ARKit-target only (hardcoded) — don't confuse them.
- `scripts/eval_recon.py` — the eval harness (open3d EGL headless): whole-scene + SUBJECT-cropped
  (median-center + 1.8×median-L1-radius) renders and stats; dihedral roughness (vectorized).
  Roughness caveat: resolution-sensitive — always judge WITH the renders. `subject_cluster.dims_mm`
  = the cleaner metric-scale check.
- `scripts/overnight_matrix.sh` — tonight's autonomous runner (§9).
- `scripts/run_pycolmap_from_record3d.py` — the owner's original preliminary script (reference recipe).

## 7. COMPUTE / VRAM FACTS (A6000 48 GB — measured, not guessed)

- fast-schedule runs: 10-16 GB — pairable.
- quality/quality_mid densification peaks (measured): face non-dense 172v @30k-sched: **47.0 GB solo
  peak** at iter~15k. ARKit feet dense @30k-sched: 34 GB at iter 9k → projected >55 GB (OOM'd, twice).
  ARKit feet non-dense @quality_mid: 16.5 GB at 15.5k (fits fine).
- **RULES:** (1) quality runs are SERIALIZED — never two in densification; (2) `mesh_extract_sdf`
  is also a GPU consumer (~10-14 GB + spikes) — it OOM'd R1 attempt 2 when paired; (3) checkpoint
  files ~0.5 GB each land in `_milo_raw/` (ckpts 8k/12k or 8k/20k); (4) disk is fine (139 GB free;
  whole program ~20 GB).
- Timing at qualmid on feet: ~2.5-3 h/run. Face fast: ~35-45 min. Face 30k-quality: ~3 h.
- The 43 GB "tripwire" pattern (background watcher that alerts before OOM) saved the program twice —
  reuse it if pairing is ever attempted again.

## 8. LESSONS LEARNED (methodology — future agents, absorb these)

1. **Cheap direct reproduction beats theorizing.** The pose-degeneracy theory fit ALL prior evidence
   and was wrong. A 30-second checkpoint-resume repro + instrumenting the actual tensors falsified
   it and found the real (resolution) cause. Always look for the fast repro before the deep theory.
2. **Make invisible settings visible.** The fast-schedule blind spot survived ~40 runs because no
   artifact recorded it. Provenance now stamps schedule+mesh_config. When adding any knob: stamp it.
3. **Fix causes at their layer.** No NaN-clamps in third-party kernels; no host-switching to dodge a
   bug; the resolution cap is enforced at the driver (the layer we own) as a documented constraint.
4. **A/Bs must share every other knob.** The fast matrix stayed valid because all arms shared the
   (wrong) schedule. The lowdens run doubled as a crash-dodge once and polluted the design — never again.
5. **Subject-cropped evaluation or nothing** — whole-scene numbers/renders are floater-dominated.
6. **Owner's instincts have been repeatedly right:** depth supervision suspicion (confirmed),
   "underutilized complex pipeline" (confirmed), refusing bandages (the resolution cause would have
   been masked by a NaN-guard). When the owner pushes back, re-examine.
7. **Serialization is near-free** when a single run saturates GPU util — pairing quality runs bought
   ~0% throughput and cost two OOMs.
8. Ops details that bit us: pgrep patterns match their own shell wrapper (verify pids via ps args
   before kill); scratchpad is session-ephemeral (durable tools live in `scripts/`); SIGSTOP does
   NOT free VRAM; background chains die with the session unless nohup+disown.
9. **Never manually pause automation without a dead-man auto-resume (2026-07-08 lesson).** The
   runner was paused for a crash diagnosis; the session then disconnected; the GPU idled ALL NIGHT.
   Rule: any pause of the autonomous runner must schedule its own bounded resume (e.g.,
   `nohup bash -c 'sleep 7200; bash scripts/overnight_matrix.sh' &`) BEFORE the diagnosis begins —
   the runner is idempotent, so a spurious resume is harmless, but a missed one costs a night.

## 9. OVERNIGHT AUTONOMOUS OPERATION (running now)

Runner: `scripts/overnight_matrix.sh`, pid in `sessions/_sweep_eval/overnight/runner.lock`,
launched 2026-07-07 19:22 detached (survives session death). Strictly serial; idempotent; self-heals
interrupted arms (labeled-output → skip; unlabeled `provenance_stage5.json` → relabel+eval; active
trainer → wait). Logs: `overnight/status.log`, per-arm `overnight/R*.log`; final artifact
`overnight/MORNING_SUMMARY.md` (all stats.json inlined).

Arms (in order), each ≈2.5-3 h:
| Arm | What | Why | Eval pair |
|---|---|---|---|
| R2' (in flight, in-session chain; runner self-heals) | HQ qualmid λ0.2 | HQ at capacity, "as intended" | vs `output_hq_dense` (schedule effect) → `_sweep_eval/hq_schedule` |
| R4' | HQ qualmid **λ0** | **THE decisive λ pair at capacity** on the SfM path | vs R2' → `hq_lambda_capacity` |
| R3' | ARKit qualmid **λ0** | λ pair on the VIO path (consistency) | vs R1'' → `arkit_lambda_capacity` |
| R6a | HQ qualmid **λ0.05** | the "whisper": floater control without noise-stamping? | vs R4' → `hq_whisper` |
| R6b | Face v3 + quality_mid | best-face demo artifact (completeness × schedule) | vs face v3-fast → `face_best` |

Failure policy: log tail → clean partial → continue to next arm (no blind retries). ETA all-done
≈ 08:00-10:00 2026-07-08.

## 10. RUNNING TASK LIST (past → present → future, with reasons and decision rules)

### DONE (with what each taught)
1. ~~HQ SfM pose recovery + LiDAR metric-lock~~ — HQ's designed path works (47/47, 1% MAD lock);
   built `03b_relock_lidar.py`.
2. ~~Root-cause the MILo crash~~ — nvdiffrast 2048 cap; `-r` cap fix; falsified the pose theory (§3.1, §8.1).
3. ~~Fast-schedule 2×2 (feet: capture × λ)~~ — λ0 smooths both paths; λ0 costs floater control.
4. ~~Component OSS-docs review~~ (5-agent) — the component map + interfaces (DAILY_NOTES 07-05).
5. ~~Face v1 (preliminary reproduction)~~ — depth-free path clean end-to-end; NO DA3 needed; 10.13°.
6. ~~Discover+fix the schedule blind spot~~ — configs/quality*, provenance stamping (§3.2-3.4).
7. ~~Face v2 (schedule A/B)~~ — modest real texture gain; distillation invariance discovered; 47 GB
   peak → serialization rule.
8. ~~Face v3 (image-set A/B)~~ — sharpness+density = visible completeness win at equal smoothness.
9. ~~R1'' (ARKit qualmid λ0.2)~~ — capacity AMPLIFIES noise-stamping (32.7°); un-gated R3'.
10. ~~Eval harness with subject crops~~ — `scripts/eval_recon.py`; ~~VRAM tripwire pattern~~;
    ~~overnight runner~~ (§9).

### IN FLIGHT (tonight, autonomous)
R2' → R4' → R3' → R6a → R6b per §9.

### MORNING DECISION RULES (apply to overnight results)
- **λ verdict at capacity:** if R4' subject-roughness ≤ R2' − 3° AND R4' subject renders coherent →
  λ0-at-capacity confirmed → architecture = anchor-only LiDAR. If R4' is smooth but SHAPELESS
  (feet melted — check `face`-style tight renders) → low-texture skin needs grounding → R6a (λ0.05)
  becomes the default candidate; compare its z-extent (floaters) and roughness vs R4'/R2'.
- **Capture-method verdict at capacity:** compare R4' (HQ) vs R3' (ARKit) — subject-mesh roughness,
  completeness in renders, subject-cluster dims sanity (feet ≈ 0.25-0.30 m within cluster; both
  arms' dims should agree — if they diverge >5-10%, flag scale work). **Tie → ARKit wins on ops**
  (no SfM step, dual anchors, live overlay); HQ wins only if visibly better.
- **Init×window test:** if R2'/R4' (sparse init, 8k window) now produce coherent feet (vs the
  fast-HQ blob), mechanism §5.4 confirmed → sparse SfM init is FINE given the window → DA3's last
  reconstruction advantage disappears.
- **R6b:** owner judges the best-face artifact vs their reference — the demo of the recommended recipe.
- **Any arm failed:** read `overnight/<arm>.log` tail; triage; the matrix tolerates single-arm loss
  (λ verdict needs R4'; capture verdict needs R4'+R3').

### QUEUED NEXT (post-matrix, in priority order)
1. **Final architecture writeup + owner recommendation** (this journal §12 draft → finalize with
   overnight data; update RESULTS/backlog/memory).
2. **Subject isolation implementation** (photometric mask + box-prune + output crop; design in
   EXPERIMENTS_BACKLOG implementation notes; owner input on crop pad welcome). Evaluate by: subject
   completeness unchanged, floaters gone, gaussian budget concentrating (subject_cluster n rising).
3. **Retention-percentile sweep** (init_cdf_mask 99% → 99.9%) if morning renders show final counts
   limit detail: the direct final-count knob.
4. **Photoreal measurement** — render trained gaussians via radegs on held-out views (PSNR/SSIM/LPIPS);
   the photoreal goal is still UNMEASURED; run on the matrix winner.
5. **Capture protocol doc + re-capture** (owner action): fill-frame, video-dense (sharpness-selected),
   ≥250 mm standoff, ruler/known-size fiducial in frame → unlocks the ruler anchor (built, unused)
   and the true 1 mm accuracy test.
6. **Vectra ground-truth protocol** (static phantom + fiducial + reference scan) — the acceptance test.
7. Optional: dense_gaussians revisit with capped window + isolation; mesh_config highres on the winner.

## 11. MORNING ANALYSIS PROTOCOL (for whoever wakes up first)

1. Read `overnight/MORNING_SUMMARY.md`; confirm all 5 arms completed (status.log).
2. Generate tight subject renders for R2'/R4'/R3'/R6a (pattern: `face_tight_AB` code in git history /
   eval_recon subject rows) — judge coherence, not just dihedral.
3. Apply §10 decision rules → fill in §12; update SWEEP_RESULTS with the four tables.
4. Owner review set: `_sweep_eval/{hq_lambda_capacity,arkit_lambda_capacity,hq_whisper,face_best}/comparison.png`
   + the winner's `.ply` files.

## 12. ARCHITECTURE RECOMMENDATION (REVISED 2026-07-08 after R4' — see SWEEP_RESULTS R4' entry)

**R4' falsified "λ0-at-capacity wins":** at capacity the λ effect is ~2°; capacity itself dominates
roughness on weak inputs (feet qualmid ≈33-35° both λ; fast-λ0 = 14.8° with VISIBLE TOES — the best
feet result). **THE LAW: capacity must match input quality.** Weak captures → fast schedule + λ0 is
their genuine optimum (starvation = implicit regularization). Strong captures (fill-frame,
video-dense) → quality schedules add real detail. λ0.2 loses everywhere; anchor-only LiDAR stands.
The pipeline therefore has an INPUT-QUALITY BRANCH, and the capture protocol is the centerpiece —
it moves captures into the branch where capacity pays.

Every strong result to date traces to: **dense pixel coverage + coherent poses + capacity matched to
input strength + no per-pixel LiDAR surface term.** Recommended pipeline:

1. **Capture:** fill the frame with the subject; video-dense frames; ≥250 mm; fiducial in frame.
   (Either app path works; ARKit preferred for ops if the matrix ties.)
2. **Frames:** sharpness-select per temporal window (v3 recipe).
3. **Poses:** from-scratch SfM (pycolmap, PINHOLE, BA-refined focal) — or ARKit VIO when present.
   DA3: optional fallback only.
4. **Metric:** anchor-based — VIO camera-path and/or LiDAR ray-median lock (03b) and/or ruler
   fiducial (best). NEVER per-pixel depth supervision for scale.
5. **Reconstruction:** MILo, quality_mid schedule (30k-stock where VRAM allows), non-dense,
   mesh reg ON, -r capped ≤2048, λ = 0 or 0.05 (tonight decides).
6. **Isolation:** subject-box photometric masking (next build) + output crop.
7. **Eval:** subject-cropped renders + dihedral + cluster-dims + (to add) radegs PSNR/LPIPS.

Open questions the overnight matrix answers: λ0 vs λ0.05 vs λ0.2 at capacity; ARKit vs HQ at
capacity; sparse-init viability. Open questions it does NOT answer: true 1 mm accuracy (needs
fiducial/Vectra), photoreal number (needs radegs eval), isolation gains (needs the build).

---
*Maintain this document: append findings with evidence + dirs; keep §10 rules current; stamp dates.*

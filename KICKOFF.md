# KICKOFF: Read this first

You are implementing a staged pipeline defined in the accompanying build specification (`00_BUILD_SPECIFICATION.md`) and its stage documents. Each stage is independent, runs in its own software environment, and communicates with the others only through files on disk, following the contracts in the `io_contracts/` folder. Before writing code, read the build specification, then these instructions.

## Target machine

The heavy stages run on a single Linux machine with:

- an NVIDIA Ampere A4000 graphics card, which has 16 gigabytes of video memory (the memory on the graphics card itself),
- 48.3 gigabytes of system memory,
- 8 processor threads.

This card supports the Compute Unified Device Architecture (CUDA) and can compile and run the graphics-card stages. Stages 2, 4, 5, and the optional Stage 3 cross-check model all run on this Linux machine.

Stage 1, the iPhone capture application, is built separately on a Mac using Xcode. The Mac has no CUDA-capable graphics hardware, so it is used only for editing and for building the capture application. It must never be used to run any graphics-card stage.

## Memory and frame budget (Stages 1 and 2)

The binding limit is the video memory of the graphics card during Stage 2, because the large front-end geometry model holds all selected views in memory at once. Therefore:

- Capture video is capped at 20 seconds per session.
- From that video, select a target of 48 keyframes for the front end, with a hard maximum of 60. This keeps the front-end model within the 16 gigabytes of video memory.
- Select keyframes by camera motion where the camera pose is available, taking a new keyframe roughly every 7.5 degrees of orbital motion, which yields even spacing regardless of how fast the clinician moves. Where the pose is unavailable, fall back to a uniform time stride that produces about the same count.
- Calibrate the true ceiling empirically on the A4000 early in the project: raise the joint view count until the video memory is nearly full, then set the cap just below that point. The figure of 48 is a well-reasoned starting target, not a measured guarantee.
- If a capture needs more coverage than the ceiling allows (for example a large body region), process the orbit in overlapping segments and stitch them together in Stage 3 using its metric anchors, or use the streaming variant of the front-end model.

## Where to start

Do not begin by building the whole pipeline at once. Begin with Stage 3 and the two week-one experiments described in Section 8 of the build specification, run on existing scans. That work needs no graphics-card compilation, no capture application, and no model porting, so it can succeed cleanly, and it proves the metric approach before anything harder is attempted. After that, follow the build order in Section 7.

## Where to expect friction (plan for a human alongside you)

Two parts of Stage 5 are the hardest and will likely require an engineer working with you rather than a hands-off result:

- Compiling the reconstruction host's older pinned toolchain from source against the graphics card. From-source graphics-card compilation is a common failure point.
- Porting the depth-and-normal supervision from a different codebase into the reconstruction host. This means translating loss-term concepts across two different code frameworks, not copying code.

Do not present these two tasks as routine scaffolding. Flag them clearly and proceed carefully.

## Ground rules

- Each stage lives in its own folder with its own isolated environment. Never merge environments.
- Stages communicate only through files, following the `io_contracts/` formats exactly.
- Run the orientation self-test whenever a coordinate convention crosses a stage boundary.

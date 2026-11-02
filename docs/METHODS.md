# Methods and reproducibility

This document describes the processing implemented in the source code rather than a literal anatomical model of an insect optic lobe.

## Sensory computation

Input is an intensity video resized by `cv2.INTER_AREA` to a height of 320 pixels, then center-cropped to a square 320 × 320 array. Motion compensation comes from frame-to-frame phase correlation and, in an additional pathway, local Farneback optical-flow residuals. Compact dark features are emphasized using a difference of narrow and wide spatial Gaussian responses (σ = 0.55 and 3.0 pixels), with a neighborhood-based texture penalty.

There are **three banks**, each containing **14 velocity-tuned activity maps**. The specific velocities in pixels per 25-fps frame are:

```text
(-1.0,-0.5) (-1.0,0.0) (-1.0,0.5)
(-0.5,-0.5) (-0.5,0.0) (-0.5,0.5)
( 0.0,-0.5)               ( 0.0,0.5)
( 0.5,-0.5) ( 0.5,0.0) ( 0.5,0.5)
( 1.0,-0.5) ( 1.0,0.0) ( 1.0,0.5)
```

Each map accumulates spatially shifted prior activity plus the current small-target evidence (leaky motion-aligned temporal integration, **not** classical Reichardt pairwise correlation): `A_v(t) = decay * shift(A_v(t-1),v) + evidence(t)`. Maps are pooled by maximum per spatial position, and the three banks plus local optical flow are normalized by no-target scale and pooled by maximum. A local-peak extractor produces up to 40 proposals, of which up to 25 are deposited into the trace memory.

## Unified trace after the velocity maps

One 320×320 memory tensor `T` is used, not one tracker per target. With previous-frame trace only:

```text
support(p) = max[Gaussian(T_previous, σ=11)[p]
                 − 1.3 × Gaussian(T_previous, σ=1.2)[p], 0]
reinforced_score(p) = sensory_strength(p) × [1 + 70 × support(p)]
T_next = 0.92 × T_previous + deposits_from_current_peaks
```

The deposit kernel is one pixel at the proposal center and 0.33 at its four axial neighbors; response weights are clipped for numerical stability. No ground-truth location is supplied to any processing stage. Current evidence updates the memory **after** the readout is computed, so there is no look-ahead. The annular filter encourages *spatial neighborhood transitions*; it does **not** explicitly compare consecutive velocity vectors. Thus it is a plausible motion-selective engineering heuristic but cannot guarantee a coherent trajectory, and random nearby detections may sometimes reinforce each other.

## Simulation and test protocol

- Full source: 365 frames at 25 fps (14.6 seconds). The detector sees intensity frames, not DVS events; the right-hand DVS-style image in the demo is derived from thresholded log-intensity changes between frames for illustration only.
- Synthetic black target: nominal 6 mm at simulated distances 0.5, 1.0, 2.0 m and assumed 84° full sensor field of view. Stamp after resize, using 12× pixel integration with `INTER_AREA` and Gaussian blur σ = 0.42 sensor pixels on the alpha mask.
- Same four full-scene trajectories for all methods/distances. Train seeds 41 and 92; test seeds 173 and 281. All results from frames 100 to 364 inclusive, 265 frames per path, 530 held-out frames per distance.
- Criterion: calculate angle between predicted and ground-truth camera rays using a pinhole camera with assumed 84° field of view; correct if within 5°. Every prediction (including low-confidence ones) counts for forced-choice localization.
- Random control: sample image positions uniformly in the eligible detector region, frames 100..364, with 300 repeated draws per path. Same random chance figure for each distance because target positions are identical.
- A separate confidence-gated analysis uses thresholds calibrated from early no-target frames. No-target false alarms must be stated alongside gated correct frames.
- Selection: memory method and hyperparameters selected only on training paths. Held-out paths use the same video scene, so this is *not* generalization to new rooms or cameras.

## Held-out scores

| Output | 0.5 m | 1 m | 2 m |
|---|---:|---:|---:|
| Raw neural peak, no extra trace | 94.2% | 89.1% | 32.5% |
| Plain low-pass memory | 93.8% | 88.9% | 32.5% |
| Individual competing hypotheses | 88.1% | 87.4% | 23.6% |
| **One shared neighborhood-memory map** | **94.0%** | **91.5%** | **39.6%** |

Original per-frame files, selected parameters and more precision are in `results/`; source needed to regenerate sensory evidence is included in the complete package.

# Methods and implementation notes

## Experimental setup

The input is a frame-based, indoor video centre-cropped to **320 × 320** at **25 frames/s**. The video has a moving camera, textured furniture, and a moving person. An artificial dark target is overlaid at an independent image position. Both target-present and target-absent versions are processed using the same detector parameters.

The diameter is set from a 6 mm target length projected into 320 sensor pixels under an approximate 84° field of view:

\[
\alpha = 2\arctan\left(\frac{L}{2d}\right), \qquad
s_{px} \simeq 320\frac{\alpha}{84^\circ}.
\]

The projected diameters are approximately **2.62 px** at 50 cm, **1.31 px** at 1 m, and **0.65 px** at 2 m. This is a simplified angular-size approximation, not a calibrated optical simulation including exact lens distortion, exposure, sensor noise or insect wing dynamics. The target is subpixel-rendered using integrated coverage and image blur.

## Motion-sensitive populations

The implementation is in `stmd_model.py` and `multi_hypothesis.py`.

1. **Dark residual**: estimate a slowly updated background, align it with the current frame via phase correlation, and emphasize pixels unexpectedly darker than the aligned background.
2. **Small-target spatial response**: difference of narrow and broad Gaussian filters on that residual.
3. **Texture-adaptive spatial inhibition**: calculate surround texture outside the receptive-field centre and reduce responses from heavily textured areas.
4. **Direction-selective temporal integration**: shift and leaky-integrate response maps along multiple candidate velocities: `A_v(t) = decay * shift(A_v(t-1), v) + evidence(t)`. This favours persistent motion consistent with a velocity.
5. **Parallel channels**: broad response, dense-motion inhibition, more texture-tolerant filtering, and an additional conventional optical-flow-registration residual channel. The last channel uses Farneback optical flow and is not a neural circuit reconstruction.

Each channel is normalized using response scales estimated from the first 100 frames of the no-target clip (99th percentile over frames 25–99). These **calibration frames are separate from the later evaluation window**, but the normalization is still scene-specific and should not be mistaken for independent generalization.

## Competing motion hypotheses

At each frame the model selects strong local maxima across the four response maps. Unmatched peaks can initiate up to **nine separate target hypotheses**, each with position, velocity, leaky support, directional consistency, and hit/miss history. Hypotheses are associated with subsequent peaks by predicted position; supported tracks gain evidence, unsupported tracks decay and disappear. The model starts with **no seeded target coordinate and no hypotheses**. This is de novo initiation with temporal priors retained afterward.

The highest-confidence track can be accepted as a detection. Confidence is a logistic, **uncalibrated heuristic** derived from neural evidence strength, persistence, direction consistency, track age, estimated motion and missed updates. It is not a true probability of insect presence.

## DVS-style visualization

The right-hand display is an **illustration of event-camera output**, not a real Dynamic Vision Sensor recording. For adjacent greyscale video frames, changes are computed in log-luminance:

\[
\Delta L_t(x,y) = \log(1+I_t(x,y)) - \log(1+I_{t-1}(x,y)).
\]

Positive changes above a display threshold are coloured green (ON-like); negative changes are magenta (OFF-like). The display is accumulated once per video frame and augmented with a faint neural-response heatmap, confidence, accepted detection, and alternative predictive tracks.

**Important:** The inference model is fed ordinary video frames and **does not operate on asynchronous GENX320 event packets**. A real event-camera implementation would need proper timestamped events, sensor bias/contrast thresholds, and latency evaluation.

## Metrics and known limitations

Synthetic ground-truth positions are used **only after inference**. A correct detection means the committed detection is within **8 pixels** of the injected dot. The main evaluation interval is frames **100–364**, following calibration on the no-target clip.

The existing benchmark records approximately **78.5%** correct at 50 cm, **60.0%** at 1 m and **0%** at 2 m over the evaluation window; the no-target false-detection fraction is approximately **2.6%**. These results concern one clip and target path, not generalized performance. At 1 m, the target is approximately 1.3 pixels across, so aliasing, local texture, motion and lighting materially affect detection. The model must not be treated as an onboard field-validated insect tracker.

The original 20-second benchmark video preserves the source **25 fps speed** but extends a **14.6-second** clip by reversing the later portion, not by slowing it down. `run_demo.py` defaults to 365 forward-only frames, to preserve the source's native continuous motion.

## Biological connections

This is best described as an **STMD-inspired hybrid system**, not a faithful full neural STMD or a reconstructed dragonfly connectome. The biology informs small-target selectivity, motion-sensitive populations, and temporal predictive facilitation. Conventional optical flow, phase correlation, threshold calibration, local maxima selection and multi-object association are engineering approximations.

See references in the main README.

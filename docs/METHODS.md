# Technical methods — original tuned two-pathway STMD

This describes the **exact implementation used for the reproduced values**. The demonstration is designed for small, dark, moving synthetic targets and is inspired by insect STMD mechanisms, without claiming an anatomically accurate full ESTMD/DSTMD circuit.

## Preprocessing

- Read original 1920×1080 video at 25 fps.
- `cv2.resize(frame, (round(width*320/height),320), interpolation=cv2.INTER_AREA)` followed by central 320×320 crop. No stretching to square.
- For the benchmark, extend the 365 original frames to 500 by reversing the last 135 frames. No temporal slowing. The primary evaluation uses frames 100–249 within original forward footage.
- For each target distance, inject a 6 mm black synthetic target after spatial downsampling. Its apparent diameter is calculated from a nominal 84° field of view as implemented in `stmd_model.apparent_diameter_px`.
- Supersample the target disk 12× and area-integrate its coverage. Gaussian smoothing of the coverage mask is **σ=0.42 pixels**, not a large image blur; this was retained unchanged to preserve the original benchmark.

## Motion-compensated dark residual

The detector maintains a background image `B` and previous gray frame `I_prev`.

```
(dx,dy)       = cv2.phaseCorrelate(I_prev, I_current)[0]
B_predicted   = warp(B, dx, dy)
residual      = max(B_predicted - I_current, 0)
response      = max(Gauss(residual, sigma=0.55) - Gauss(residual, sigma=3.0), 0)
B_next        = 0.976 * B_predicted + 0.024 * I_current
```

The phase-correlation step is conventional computer vision; the Gaussian center/surround-like response emphasizes localized dark deviations. This **does not** reproduce photoreceptor-level ON/OFF synapses.

## Texture-sensitive surround inhibition

The detector estimates the mean and standard deviation of present-frame intensities in a Gaussian surround of sigma 3 px while excluding a radius around each central receptive field. It then forms:

```
evidence = response * exp(-surround_std / 7.0)
```

The tuned model has **two parallel populations**:

- **Broad:** this texture-suppressed evidence, lightly smoothed at sigma 0.35 px.
- **Selective:** additionally multiplies by `exp(-16.0 * motion_density)`, where motion density is a Gaussian-smoothed binary residual mask (`residual > 10` intensity units).

## Directional motion populations and temporal facilitation

For each of 14 motion velocities in the grid `vx in {-1,-0.5,0,.5,1}` and `vy in {-0.5,0,.5}`, excluding `(0,0)`, the response is propagated using a spatial shift and then integrated:

```
A_v(t) = 0.85 * warp(A_v(t-1), v_x, v_y) + evidence(t)
```

The model computes the maximum across channels at each pixel, suppresses a 12-pixel border, and reads the global maximum and its associated velocity. These are **motion-aligned recurrent integration populations** (not delayed input multipliers or a trained neural network). In particular, there are **no competing trajectories/hypothesis objects** or a target identity tracker.

## Thresholds and output

Calibrate two scalar global thresholds using a **dot-free control**: the 99th percentile of each pathway's maximal activity over frames 25–99, clipped below at 1e-6. Thresholds are fixed for all target distances.

At each frame, divide each pathway's global maximum by its threshold, choose the pathway with largest normalized activity, and accept if this normalized score is at least 1. The exported **confidence** is a logistic mapping `sigmoid(4*(normalized_score-1))`; it is **not a calibrated probability**.

A correct accepted detection must fall within 8 pixels of the synthetic target coordinate. The coordinate exists only for target injection and evaluation, not for detector inference.

## Results and validity

Exactly reproduced, original 1920×1080 footage, 25 fps, frames 100–249:

| distance | apparent size | correct |
|---|---:|---:|
|50 cm|2.62 px|88.0%|
|1 m|1.31 px|76.7%|
|2 m|0.65 px|33.3%|

No-target false alarms in that 150-frame interval: 0.0%; across 500 frames: 11.6%.

These results are **trajectory- and scene-specific**, not an unbiased validation of live mosquito detection or performance on a physical DVS camera. Threshold calibration is performed on the same background scene as evaluation. The source camera's physical field of view was not established. Video compression and synthetic dot drawing parameters materially affect subpixel performance.

## DVS visualization (not detector input)

The side-by-side HD demo renders one 320×320 image per conventional video frame by differencing *log(1+gray)* values:

```
d = log1p(I_current) - log1p(I_previous)
ON  = max(d - 0.045, 0)
OFF = max(-d - 0.045, 0)
```

The ON and OFF values are contrast-expanded for display. Green=ON, magenta=OFF; a faint network response heatmap and a readout glyph are overlaid. It is **not asynchronous DVS data with event timestamps**, and detector accuracy numbers refer to the conventional intensity-frame model, not to a DVS-driven algorithm.

## Reference papers

- Wiederman, Shoemaker & O'Carroll (2013), DOI: https://doi.org/10.1523/JNEUROSCI.1277-13.2013
- Wang, Peng & Yue (2020), DOI: https://doi.org/10.1109/TCYB.2018.2869384
- Wang, Peng, Zheng & Yue (2020), DOI: https://doi.org/10.1109/TNNLS.2019.2910418
- Wiederman et al. (2017), DOI: https://doi.org/10.7554/eLife.26478

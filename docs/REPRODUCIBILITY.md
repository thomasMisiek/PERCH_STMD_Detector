# Reproducibility and important corrections

- The 365-frame (14.6 s) source is real household footage; synthetic 6 mm target is inserted after 320×320 area downsampling, using the same full-scene randomized paths across all distances and conditions.
- The `DVS` visual panel is frame-to-frame log-intensity change display, NOT input to the STMD and NOT output from a hardware DVS. The STMD intensity-based dark-target response is separate.
- A previous architecture animation mistakenly took one 2-D activity map and treated its rows as distinct velocity maps. The new `src/render_process.py` asserts the stack has shape (24, 320, 320) and visualizes the real recurrent arrays.
- The signed map display is `A_k - mean_k(A_k)` on a crop centered on the strongest sensory local maximum, chosen without GT. This is a VIEW, not model computation. The original recurrent responses are used for fusion.
- Previous still-background calibration produced scales near 0 while the moving source produced approximately [48.27, 17.01, 49.34, 64.50]. The earlier extremely poor still-background accuracy was caused by incomparable near-zero calibration. This project calibrates all speed conditions on the same no-target moving clip and reports corrected results.
- The 24-channel bank was selected against the smaller grid using a previous memory stage; comparative 24 vs smaller bank scores are not claimed as a new predictive-field ablation.
- The normalized predictive field has bounded 0..1 spatial activity and a learned-from-detections velocity prediction. It is not an anatomically specified continuous neural attractor.
- Confidence is not calibrated as P(target). Results report forced-choice angular localization within 5°, not precision/recall of confidently accepted real insects.
- The predictor does not use the true target coordinates; red input ring and accuracy figures use GT only after inference.
- Previous inhibitory-ring memory is kept as an explicit *comparison* baseline. The selected executable uses `src/predictive_field.py`, not it.
- Code and demos show one bank's 24 maps; all three banks contain 24 motion maps (72 total), plus one local-flow branch.

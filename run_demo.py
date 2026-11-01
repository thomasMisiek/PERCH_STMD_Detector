"""Run the PERCH STMD-inspired demo on a real video, with optional synthetic small target.

Examples:
    python run_demo.py --distance 1m
    python run_demo.py --distance 50cm --frames 250
    python run_demo.py --all --frames 500
    python run_demo.py --input path/to/your_video.mp4 --distance 1m

Ground truth is used strictly for post-hoc evaluation, never detection.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import multi_hypothesis as model
from render_and_report import render, calculate_metrics

ROOT = Path(__file__).resolve().parent


def load_video(path: Path, n: int):
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        neww = round(w * model.SIDE / h)
        frame = cv2.resize(frame, (neww, model.SIDE), interpolation=cv2.INTER_AREA)
        if neww < model.SIDE:
            raise ValueError('Input video too narrow for a square center crop')
        x0 = (neww - model.SIDE) // 2
        frames.append(frame[:, x0:x0+model.SIDE])
    cap.release()
    if not frames:
        raise RuntimeError('No frames read from input video')
    # If n is greater than source duration, extend using reverse playback, never slow motion.
    ids = list(range(len(frames)))
    if n > len(ids):
        reverse = list(range(len(frames)-2, -1, -1))
        while len(ids) < n:
            ids.extend(reverse)
    return np.asarray(frames, dtype=np.uint8)[ids[:n]], float(fps)


def main():
    p = argparse.ArgumentParser(description='Small-target motion demo with competing hypotheses')
    p.add_argument('--input', type=Path, default=ROOT/'data'/'source_indoor_walking.mp4', help='Input household video')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--distance', choices=['50cm','1m','2m'], default='1m')
    group.add_argument('--all', action='store_true', help='Render all distances')
    p.add_argument('--frames', type=int, default=365, help='Number of output frames (365=original 14.6s; 500=20s forward/reverse)')
    p.add_argument('--out', type=Path, default=ROOT/'outputs', help='Output directory')
    args = p.parse_args()
    if args.frames < 100:
        p.error('--frames should be at least 100 to permit initial no-target calibration')
    args.out.mkdir(parents=True, exist_ok=True)

    base, fps = load_video(args.input, args.frames)
    if abs(fps-25.0) > 1e-4:
        print(f'Warning: source is {fps:.2f} fps; tracking time constants were tuned at 25 fps.', file=sys.stderr)
    print(f'Read {len(base)} frames at {fps:.2f} fps ({len(base)/fps:.2f} seconds)')
    scales = model.calibrate(base[:100]); print('No-target evidence scales:', scales)
    no_df, no_visuals = model.run(base, scales, with_visuals=True)
    no_df.to_csv(args.out/'no_target_results.csv', index=False)
    cases = model.DISTANCES if args.all else [next(x for x in model.DISTANCES if x[1] == args.distance)]
    summary = [{'condition':'no target', **calculate_metrics(no_df)}]
    for distance, label in cases:
        with_dot, gt, apparent = model.inject(base, distance)
        result, visual = model.run(with_dot, scales, gt, with_visuals=True)
        result.to_csv(args.out/f'{label}_results.csv', index=False)
        render(with_dot, visual, base, no_visuals, args.out/f'STMD_{label}.mp4', label)
        summary.append({'condition':label, 'diameter_px':apparent, **calculate_metrics(result, gt)})
        print(f'{label}: {apparent:.2f}px projected target, ' 
              f'evaluation correct {result.correct.iloc[100:min(365,len(result))].mean():.1%}')
    pd.DataFrame(summary).to_csv(args.out/'summary.csv', index=False)
    print(f'Finished: {args.out}')

if __name__ == '__main__':
    main()

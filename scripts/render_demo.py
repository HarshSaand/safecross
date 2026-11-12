#!/usr/bin/env python3
"""Render the SafeCross baseline comparison as MP4 or GIF."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safecross.visualization import generate_reference_scenario, render_comparison_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "safecross_demo.mp4")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--frame-stride", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenario = generate_reference_scenario()
    rendered = render_comparison_video(
        scenario,
        args.output,
        fps=args.fps,
        dpi=args.dpi,
        frame_stride=args.frame_stride,
    )
    print(f"Rendered SafeCross demo: {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


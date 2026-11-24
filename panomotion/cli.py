"""Command line interface - auxiliary to the graphical player.

The primary entry point for this tool is the graphical motion analysis
player::

    python -m panomotion gui

The CLI only supports the spherical workflow: generating the known-motion
synthetic panorama sequence, analysing an existing equirectangular frame
folder, and launching the player.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import numpy as np

from .pipeline import PipelineConfig, run_pipeline

DEFAULT_RUNS = Path.cwd() / "runs"


def cmd_synth(args):
    from .synth import generate_synthetic
    out = Path(args.runs_dir) / args.name
    seq = generate_synthetic(
        width=args.width, height=args.height, n_frames=args.frames,
        seed=args.seed)
    pcfg = PipelineConfig(name=args.name, cols=args.cols, rows=args.rows,
                          n_frames=args.frames, width=args.width,
                          height=args.height, run_baseline=not args.no_baseline)
    run_pipeline(seq.frames, out, seq=seq, pcfg=pcfg,
                 progress=lambda s, f, m: print(f"[{s}] {m}", flush=True))
    print(f"\nDone. Open the player with: python -m panomotion gui\n"
          f"Run directory: {out}")


def load_frames(input_path: Path):
    if input_path.is_dir():
        files = sorted([p for p in input_path.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
        if not files:
            raise SystemExit(f"no images found in {input_path}")
        from .media import load_rgb
        return [load_rgb(p) for p in files]
    # Video file via imageio + bundled ffmpeg.
    import imageio.v2 as imageio
    reader = imageio.get_reader(str(input_path))
    frames = [np.asarray(im)[..., :3] for im in reader]
    reader.close()
    if not frames:
        raise SystemExit(f"could not decode video {input_path}")
    return frames


def cmd_analyze(args):
    src = Path(args.input)
    frames = load_frames(src)
    if args.max_frames:
        frames = frames[:args.max_frames]
    h, w = frames[0].shape[:2]
    out = Path(args.runs_dir) / args.name
    pcfg = PipelineConfig(name=args.name, cols=args.cols, rows=args.rows,
                          width=w, height=h, run_baseline=False)
    run_pipeline(frames, out, seq=None, pcfg=pcfg,
                 progress=lambda s, f, m: print(f"[{s}] {m}", flush=True))
    print(f"Done: {out}")


def cmd_gui(args):
    from .server import serve
    url = f"http://{args.host}:{args.port}/"
    print(f"Panoramic motion analysis player: {url}")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    serve(host=args.host, port=args.port, runs_dir=Path(args.runs_dir))


def build_parser():
    p = argparse.ArgumentParser(prog="panomotion",
                                description="Panoramic spherical motion tool")
    sub = p.add_subparsers(dest="command")

    g = sub.add_parser("gui", help="launch the graphical motion analysis player")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8765)
    g.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    g.add_argument("--no-browser", action="store_true")
    g.set_defaults(func=cmd_gui)

    s = sub.add_parser("synth", help="generate synthetic GT sequence and analyse")
    s.add_argument("--name", default="demo")
    s.add_argument("--frames", type=int, default=12)
    s.add_argument("--width", type=int, default=960)
    s.add_argument("--height", type=int, default=480)
    s.add_argument("--cols", type=int, default=64)
    s.add_argument("--rows", type=int, default=32)
    s.add_argument("--seed", type=int, default=11)
    s.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    s.add_argument("--no-baseline", action="store_true")
    s.set_defaults(func=cmd_synth)

    a = sub.add_parser("analyze", help="analyse an equirect frame folder/video")
    a.add_argument("input")
    a.add_argument("--name", default="custom")
    a.add_argument("--cols", type=int, default=64)
    a.add_argument("--rows", type=int, default=32)
    a.add_argument("--max-frames", type=int, default=0)
    a.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    a.set_defaults(func=cmd_analyze)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        # Graphical entry is the default behaviour.
        args = parser.parse_args(["gui"] + (argv or []))
        if not getattr(args, "func", None):
            parser.print_help()
            return 0
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

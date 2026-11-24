"""Dependency-free local web server for the spherical motion player.

Serves the static frontend, run artifacts (frames / videos / JSON) and a
small JSON API.  A background job can regenerate the known-motion demo.
"""

from __future__ import annotations

import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WEB_DIR = Path(__file__).resolve().parent / "web"

JOB_LOCK = threading.Lock()
JOB = {"running": False, "stage": "idle", "frac": 0.0,
       "message": "", "error": None, "done": False}


def _update(stage, frac, message):
    with JOB_LOCK:
        JOB.update(running=True, stage=stage, frac=float(frac),
                   message=message, error=None, done=False)


def _build_demo(runs_dir: Path, frames: int, width: int, height: int):
    try:
        from .synth import generate_synthetic
        from .pipeline import PipelineConfig, run_pipeline
        _update("synth", 0.02, "合成已知相机旋转与局部物体运动的全景序列")
        seq = generate_synthetic(width=width, height=height,
                                 n_frames=frames, verbose=False)
        pcfg = PipelineConfig(name="demo", cols=64, rows=32,
                              n_frames=frames, width=width, height=height)

        def prog(stage, frac, msg):
            _update(stage, 0.05 + 0.95 * frac, msg)

        run_pipeline(seq.frames, runs_dir / "demo", seq=seq, pcfg=pcfg,
                     progress=prog)
        with JOB_LOCK:
            JOB.update(running=False, stage="done", frac=1.0,
                       message="分析完成", done=True, error=None)
    except Exception as e:  # surface in the UI
        traceback.print_exc()
        with JOB_LOCK:
            JOB.update(running=False, stage="error",
                        message="构建失败", error=f"{type(e).__name__}: {e}")


class Handler(BaseHTTPRequestHandler):
    runs_dir: Path = Path("runs")

    def log_message(self, fmt, *args):
        pass

    # -- helpers -----------------------------------------------------------
    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path):
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".mp4": "video/mp4", ".webm": "video/webm",
            ".svg": "image/svg+xml", ".ico": "image/x-icon",
        }.get(path.suffix.lower(), "application/octet-stream")
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if rng and rng.startswith("bytes="):
            try:
                a, b = rng[6:].split("-", 1)
                start = int(a) if a else 0
                end = int(b) if b else size - 1
                end = min(end, size - 1)
                partial = start <= end
            except ValueError:
                partial = False
        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _run_path(self, name, *parts):
        root = (self.runs_dir / name).resolve()
        p = root.joinpath(*parts).resolve()
        if root not in p.parents and p != root:
            return None
        return p

    # -- routing -----------------------------------------------------------
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        q = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            return self._file(WEB_DIR / "index.html")
        if path == "/app.js":
            return self._file(WEB_DIR / "app.js")
        if path == "/styles.css":
            return self._file(WEB_DIR / "styles.css")

        if path == "/api/runs":
            return self.api_runs()
        if path == "/api/job":
            with JOB_LOCK:
                return self._json(dict(JOB))
        if path == "/api/report":
            return self.api_simple(q, "report.json")
        if path == "/api/meta":
            return self.api_simple(q, "meta.json")
        if path == "/api/tracks":
            return self.api_simple(q, "tracks.json")
        if path == "/api/pair":
            name = q.get("name", ["demo"])[0]
            i = int(q.get("i", ["0"])[0])
            p = self._run_path(name, "data", f"pair_{i:03d}.json")
            if p is None:
                return self.send_error(403)
            return self._file(p)

        if path.startswith("/artifacts/"):
            rel = path[len("/artifacts/"):]
            parts = rel.split("/", 1)
            if len(parts) != 2:
                return self.send_error(404)
            name, sub = parts
            p = self._run_path(name, *sub.split("/"))
            if p is None:
                return self.send_error(403)
            return self._file(p)

        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/build":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        with JOB_LOCK:
            if JOB["running"]:
                return self._json({"started": False, "reason": "already running"})
            JOB.update(running=True, stage="queued", frac=0.0,
                        message="已排队", error=None, done=False)
        frames = int(body.get("frames", 12))
        width = int(body.get("width", 960))
        height = width // 2
        t = threading.Thread(target=_build_demo,
                             args=(self.runs_dir, frames, width, height),
                             daemon=True)
        t.start()
        self._json({"started": True})

    def api_runs(self):
        runs = []
        if self.runs_dir.exists():
            for d in sorted(self.runs_dir.iterdir()):
                if d.is_dir() and (d / "meta.json").exists():
                    try:
                        meta = json.loads((d / "meta.json").read_text("utf-8"))
                        has_report = (d / "report.json").exists()
                        runs.append({"name": d.name, "meta": meta,
                                     "has_report": has_report})
                    except Exception:
                        pass
        self._json({"runs": runs})

    def api_simple(self, q, fname):
        name = q.get("name", ["demo"])[0]
        p = self._run_path(name, fname)
        if p is None or not p.exists():
            return self._json({"error": f"{fname} not found for run {name}"}, 404)
        return self._file(p)


def serve(host: str = "127.0.0.1", port: int = 8765,
          runs_dir: Path | str = "runs"):
    runs_dir = Path(runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    class _H(Handler):
        pass

    _H.runs_dir = runs_dir
    httpd = ThreadingHTTPServer((host, port), _H)
    print(f"Serving UI on http://{host}:{port}  (runs: {runs_dir})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()

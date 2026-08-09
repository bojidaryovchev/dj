"""
DJ Crate helper.

Local job server that the Chrome extension talks to. Owns the yt-dlp process
pool so that downloads survive the extension's service worker being torn down.

Stdlib only -- no pip install required.

    python server.py
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
COOKIE_DIR = HERE / "cookies"
ARCHIVE_PATH = HERE / "archive.txt"
BIN = HERE / "bin"                      # populated by bootstrap.py

# yt-dlp emits one of these per progress tick; see --progress-template below.
PROGRESS_RE = re.compile(r"^PROG\|([^|]*)\|([^|]*)\|(.*)$")
PERCENT_RE = re.compile(r"([\d.]+)")


def resolve_tool(name: str, configured: str | None = None) -> str:
    """
    Locate an executable, preferring our own bin/ over the system.

    Order: bin/ (bootstrap.py) -> explicit config path -> PATH. This keeps a
    machine with no yt-dlp installed working, and pins us to a known-good
    ffmpeg rather than whatever old build happens to be first on PATH.
    """
    local = BIN / (name if name.endswith(".exe") else f"{name}.exe")
    if local.exists():
        return str(local)
    if configured and configured != name:
        return configured
    return shutil.which(name) or ""


def load_config() -> dict:
    # utf-8-sig, not utf-8: tolerate a BOM from PowerShell or Notepad rather
    # than dying on it.
    with CONFIG_PATH.open(encoding="utf-8-sig") as fh:
        return json.load(fh)


def save_config(cfg: dict) -> None:
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    tmp.replace(CONFIG_PATH)


class Job:
    __slots__ = ("id", "url", "title", "status", "percent", "speed", "eta",
                 "message", "filepath", "created")

    def __init__(self, url: str, title: str):
        self.id = uuid.uuid4().hex[:12]
        self.url = url
        self.title = title
        self.status = "queued"   # queued | running | done | error | skipped
        self.percent = 0.0
        self.speed = ""
        self.eta = ""
        self.message = ""
        self.filepath = ""
        self.created = time.time()

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [self._jobs[i].as_dict() for i in self._order]

    def clear_finished(self) -> int:
        with self._lock:
            keep = [i for i in self._order
                    if self._jobs[i].status in ("queued", "running")]
            removed = len(self._order) - len(keep)
            self._jobs = {i: self._jobs[i] for i in keep}
            self._order = keep
            return removed


STORE = JobStore()
WORK_Q: "queue.Queue[Job]" = queue.Queue()


def cookie_file_for(url: str) -> Path | None:
    """Return the cookie jar the extension exported for this URL's site, if any."""
    host = "soundcloud" if "soundcloud.com" in url else "youtube"
    path = COOKIE_DIR / f"{host}.txt"
    return path if path.exists() and path.stat().st_size > 0 else None


def build_command(job: Job, cfg: dict) -> list[str]:
    out_dir = Path(cfg["download_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ytdlp = resolve_tool("yt-dlp", cfg.get("ytdlp_path"))

    cmd = [
        ytdlp or "yt-dlp",
        "--no-playlist",
        "--newline",
        "--no-simulate",
        # --print implies quiet, which would swallow the progress template;
        # --progress forces it back on.
        "--progress",
        "--print", "after_move:filepath",
        "--progress-template",
        "download:PROG|%(progress._percent_str)s|%(progress._speed_str)s|%(progress._eta_str)s",
        # Prefer a container we can remux rather than re-encode where possible.
        "-f", "bestaudio[ext=m4a]/bestaudio/best",
        "-x",
        "--audio-format", cfg.get("audio_format", "mp3"),
        "--audio-quality", str(cfg.get("audio_quality", "0")),
        "--embed-metadata",
        "--embed-thumbnail",
        # Split "Artist - Title" video titles into real ID3 fields so the crate
        # doesn't import as 40 tracks called "(Official Video) [HD]".
        "--parse-metadata", "title:%(artist)s - %(title)s",
        "-o", os.path.join(str(out_dir),
                           "%(artist,uploader)s - %(track,title)s.%(ext)s"),
        "--retries", "3",
        "--fragment-retries", "10",
    ]

    # Point yt-dlp at our bundled ffmpeg when we have one, so the transcode
    # doesn't fall back to an unknown system build.
    if (BIN / "ffmpeg.exe").exists():
        cmd += ["--ffmpeg-location", str(BIN)]

    if cfg.get("use_archive", True):
        cmd += ["--download-archive", str(ARCHIVE_PATH)]

    if cfg.get("use_cookies", True):
        jar = cookie_file_for(job.url)
        if jar:
            cmd += ["--cookies", str(jar)]

    extra = cfg.get("extra_args") or []
    if isinstance(extra, list):
        cmd += [str(a) for a in extra]

    cmd.append(job.url)
    return cmd


def run_job(job: Job, cfg: dict) -> None:
    job.status = "running"
    cmd = build_command(job, cfg)

    creation = 0
    if sys.platform == "win32":
        creation = subprocess.CREATE_NO_WINDOW

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation,
        )
    except FileNotFoundError:
        job.status = "error"
        job.message = "yt-dlp not found — run: python bootstrap.py"
        return

    tail: list[str] = []
    already_downloaded = False

    for line in proc.stdout:
        line = line.rstrip("\n")
        if not line:
            continue

        m = PROGRESS_RE.match(line)
        if m:
            pct, speed, eta = m.groups()
            num = PERCENT_RE.search(pct)
            if num:
                job.percent = float(num.group(1))
            job.speed = speed.strip()
            job.eta = eta.strip()
            continue

        if "has already been recorded in the archive" in line:
            already_downloaded = True

        # Post-processing runs after the bar hits 100%; say so rather than
        # sitting at "100%" for the length of an ffmpeg transcode.
        if line.startswith("[ExtractAudio]"):
            job.message = "converting"
        elif line.startswith(("[Metadata]", "[EmbedThumbnail]", "[ThumbnailsConvertor]")):
            job.message = "tagging"

        # --print after_move:filepath writes the final path on its own line.
        if os.path.isabs(line) and Path(line).parent.exists():
            job.filepath = line

        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)

    code = proc.wait()

    if code == 0 and already_downloaded and not job.filepath:
        job.status = "skipped"
        job.message = "Already in archive"
    elif code == 0:
        job.status = "done"
        job.percent = 100.0
        job.message = os.path.basename(job.filepath) if job.filepath else "Done"
    else:
        job.status = "error"
        errs = [t for t in tail if "ERROR" in t or "Unsupported" in t]
        job.message = (errs[-1] if errs else
                       (tail[-1] if tail else f"yt-dlp exited {code}"))


def worker_loop() -> None:
    while True:
        job = WORK_Q.get()
        try:
            run_job(job, load_config())
        except Exception as exc:                      # keep the pool alive
            job.status = "error"
            job.message = f"{type(exc).__name__}: {exc}"
        finally:
            WORK_Q.task_done()


class Handler(BaseHTTPRequestHandler):
    server_version = "DJCrateHelper/1.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin", "")
        allowed = load_config().get("extension_id", "")
        return origin == f"chrome-extension://{allowed}"

    def _cors(self) -> None:
        allowed = load_config().get("extension_id", "")
        self.send_header("Access-Control-Allow-Origin",
                         f"chrome-extension://{allowed}")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Auth-Token")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _authed(self) -> bool:
        if not self._origin_ok():
            self._send(403, {"error": "origin not allowed"})
            return False
        if self.headers.get("X-Auth-Token") != load_config().get("token"):
            self._send(401, {"error": "bad token"})
            return False
        return True

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # -- routes -----------------------------------------------------------

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            cfg = load_config()
            self._send(200, {
                "ok": True,
                "download_dir": cfg["download_dir"],
                "concurrency": cfg.get("concurrency", 3),
                "audio_format": cfg.get("audio_format", "mp3"),
                "ytdlp": resolve_tool("yt-dlp", cfg.get("ytdlp_path")),
                "ffmpeg": resolve_tool("ffmpeg"),
                # yt-dlp has deprecated YouTube extraction without a JS
                # runtime; without one, some formats silently go missing.
                "js_runtime": shutil.which("deno") or shutil.which("node") or "",
                "cookies": sorted(p.stem for p in COOKIE_DIR.glob("*.txt")
                                  if p.stat().st_size > 0),
            })
            return

        if self.path == "/jobs":
            if not self._authed():
                return
            self._send(200, {"jobs": STORE.snapshot()})
            return

        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._authed():
            return

        data = self._body()

        if self.path == "/jobs":
            items = data.get("items") or []
            created = []
            for item in items:
                url = (item.get("url") or "").strip()
                if not url:
                    continue
                job = Job(url, (item.get("title") or url).strip())
                STORE.add(job)
                WORK_Q.put(job)
                created.append(job.as_dict())
            self._send(200, {"created": created})
            return

        if self.path == "/cookies":
            # The extension reads cookies via chrome.cookies and posts them
            # here. This is what lets us work at all on Windows, where Chrome's
            # app-bound encryption blocks --cookies-from-browser.
            COOKIE_DIR.mkdir(parents=True, exist_ok=True)
            written = []
            for host, text in (data.get("jars") or {}).items():
                if host not in ("youtube", "soundcloud"):
                    continue
                path = COOKIE_DIR / f"{host}.txt"
                path.write_text(text, encoding="utf-8")
                try:
                    os.chmod(path, 0o600)
                except OSError:
                    pass
                written.append(host)
            self._send(200, {"written": written})
            return

        if self.path == "/config":
            cfg = load_config()
            for key in ("download_dir", "concurrency", "audio_format",
                        "audio_quality", "use_cookies", "use_archive"):
                if key in data:
                    cfg[key] = data[key]
            save_config(cfg)
            self._send(200, {"config": cfg})
            return

        if self.path == "/clear":
            self._send(200, {"removed": STORE.clear_finished()})
            return

        self._send(404, {"error": "not found"})


def main() -> None:
    if not CONFIG_PATH.exists():
        sys.exit(f"Missing {CONFIG_PATH}. Copy config.example.json to config.json.")

    cfg = load_config()
    COOKIE_DIR.mkdir(parents=True, exist_ok=True)

    workers = int(cfg.get("concurrency", 3))
    for _ in range(workers):
        threading.Thread(target=worker_loop, daemon=True).start()

    port = int(cfg.get("port", 8765))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)

    print(f"DJ Crate helper listening on http://127.0.0.1:{port}")
    print(f"  download dir : {cfg['download_dir']}")
    print(f"  workers      : {workers}")
    print(f"  extension    : chrome-extension://{cfg.get('extension_id')}")
    ytdlp = resolve_tool("yt-dlp", cfg.get("ytdlp_path"))
    ffmpeg = resolve_tool("ffmpeg")
    js = shutil.which("deno") or shutil.which("node")

    print(f"  yt-dlp       : {ytdlp or 'NOT FOUND -- run: python bootstrap.py'}")
    print(f"  ffmpeg       : {ffmpeg or 'NOT FOUND -- run: python bootstrap.py'}")
    print(f"  js runtime   : {js or 'NOT FOUND -- some YouTube formats will be missing'}")
    print("Ctrl+C to stop.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()

"""
DJ Crate helper.

Local job server that the Chrome extension talks to. Owns the yt-dlp process
pool so that downloads survive the extension's service worker being torn down.

Stdlib only -- no pip install required.

    python server.py
"""

from __future__ import annotations

import hmac
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
LIBRARY_PATH = HERE / "library.json"
BIN = HERE / "bin"                      # populated by bootstrap.py

# yt-dlp emits one of these per progress tick; see --progress-template below.
PROGRESS_RE = re.compile(r"^PROG\|([^|]*)\|([^|]*)\|(.*)$")
PERCENT_RE = re.compile(r"([\d.]+)")

# ...and one of these per post-processing stage. We ask for these explicitly
# because --print implies --quiet, which silences yt-dlp's ordinary
# "[ExtractAudio] Destination: ..." chatter -- leaving the bar parked at 100%
# for the whole transcode, which on an hour-long set looks like a hang.
PP_RE = re.compile(r"^PP\|([^|]*)\|(.*)$")

PP_LABELS = {
    "MetadataParser": "reading tags",
    "ExtractAudio": "converting",
    "Metadata": "tagging",
    "EmbedThumbnail": "artwork",
    "ThumbnailsConvertor": "artwork",
    "MoveFiles": "filing",
}


def pp_label(name: str) -> str:
    if name.startswith("Fixup"):
        return "remuxing"
    return PP_LABELS.get(name, "processing")


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
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def save_config(cfg: dict) -> None:
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    tmp.replace(CONFIG_PATH)


# -- library ---------------------------------------------------------------
#
# What we've actually got, url -> where it landed. This replaces yt-dlp's
# --download-archive, which records only "youtube <id>" and so can never
# notice that you moved, renamed or deleted the file: it just keeps claiming
# you have a track that isn't on disk any more. Storing the path lets every
# skip be re-checked against the filesystem.

LIBRARY_LOCK = threading.Lock()


def load_library() -> dict:
    try:
        with LIBRARY_PATH.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def have_file(url: str, library: dict | None = None) -> str:
    """The file we hold for this URL, or "" if it's gone or we never had it."""
    rec = (load_library() if library is None else library).get(url)
    path = (rec or {}).get("path", "")
    return path if path and os.path.exists(path) else ""


def record_download(job: "Job") -> None:
    with LIBRARY_LOCK:
        library = load_library()
        library[job.url] = {
            "path": job.filepath,
            "title": job.title,
            "bytes": os.path.getsize(job.filepath) if os.path.exists(job.filepath) else 0,
            "at": time.time(),
        }
        tmp = LIBRARY_PATH.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(library, fh, indent=2, ensure_ascii=False)
        tmp.replace(LIBRARY_PATH)


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
        "--progress-template",
        "postprocess:PP|%(progress.status)s|%(progress.postprocessor)s",
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
        # Second line of defence behind the library check: if a track resolves
        # to a name we already hold, leave the existing file alone.
        "--no-overwrites",
    ]

    # Point yt-dlp at our bundled ffmpeg when we have one, so the transcode
    # doesn't fall back to an unknown system build.
    if (BIN / "ffmpeg.exe").exists():
        cmd += ["--ffmpeg-location", str(BIN)]

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
    # Settle this against the filesystem before touching the network: if the
    # file we recorded is still sitting there, there is nothing to do.
    if cfg.get("use_archive", True):
        held = have_file(job.url)
        if held:
            job.status = "skipped"
            job.filepath = held
            job.message = f"already have {os.path.basename(held)}"
            return

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

        # Name the ffmpeg stage that's running, so the wait after the bar
        # fills reads as work rather than as a stall.
        m = PP_RE.match(line)
        if m:
            status, name = m.groups()
            if status == "started":
                job.percent = 100.0
                job.speed = job.eta = ""
                job.message = pp_label(name)
            continue

        # --print after_move:filepath writes the final path on its own line.
        if os.path.isabs(line) and Path(line).parent.exists():
            job.filepath = line

        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)

    code = proc.wait()

    # A real download always announces its destination via --print
    # after_move:filepath, so a clean exit with nothing printed means yt-dlp
    # skipped the track -- practically always the download archive. Don't try
    # to read its "already recorded" notice instead: --print implies --quiet,
    # which swallows that line, and --progress only restores the progress bar.
    if code == 0 and not job.filepath:
        job.status = "skipped"
        job.message = "file already on disk — nothing downloaded"
    elif code == 0:
        job.status = "done"
        job.percent = 100.0
        job.message = os.path.basename(job.filepath)
        record_download(job)
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
        """
        Reject callers from another origin.

        A missing Origin is normal, not suspicious: our host_permissions entry
        makes Chrome skip CORS for these requests, and without cors mode it
        only attaches Origin when the method isn't GET/HEAD -- so every
        GET /jobs arrives without one. Nothing that can omit the header can
        also set X-Auth-Token (that forces cors mode, which forces Origin),
        so the token check below is what gates those.
        """
        origin = self.headers.get("Origin")
        if origin is None:
            return True
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

    def _drain(self) -> None:
        """
        Consume a request body we're about to reject unread.

        We speak HTTP/1.1, so the connection is reused. Bytes left in the
        socket get parsed as the next request line -- a 401 on /jobs would
        otherwise turn the following /cookies POST into a bogus 400.
        """
        remaining = int(self.headers.get("Content-Length") or 0)
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 65536))
            if not chunk:
                break
            remaining -= len(chunk)

    def _authed(self) -> bool:
        if not self._origin_ok():
            self._drain()
            self._send(403, {"error": "origin not allowed"})
            return False
        if not hmac.compare_digest(self.headers.get("X-Auth-Token") or "",
                                   load_config().get("token") or ""):
            self._drain()
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

        if self.path == "/library":
            if not self._authed():
                return
            # Verified against disk on every call, so a track you deleted
            # stops being reported as held and becomes downloadable again.
            library = load_library()
            self._send(200, {"have": {
                url: os.path.basename(rec["path"])
                for url, rec in library.items()
                if have_file(url, library)
            }})
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

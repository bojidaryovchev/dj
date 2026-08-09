"""
Fetch yt-dlp and ffmpeg into helper/bin/ so the helper doesn't depend on
whatever happens to be on PATH.

    python bootstrap.py            # get whatever is missing
    python bootstrap.py --force    # re-download everything (also how you update)

Stdlib only. Windows x64.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

BIN = Path(__file__).resolve().parent / "bin"

YTDLP_URL = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe"

# yt-dlp publishes its own ffmpeg builds; it recommends these over stock
# releases because upstream ships AAC/HLS bugs that affect extraction.
FFMPEG_URL = ("https://github.com/yt-dlp/FFmpeg-Builds/releases/download/"
              "latest/ffmpeg-master-latest-win64-gpl.zip")

WANTED_FROM_ZIP = ("ffmpeg.exe", "ffprobe.exe")


def download(url: str, dest: Path, label: str) -> None:
    print(f"  {label}: downloading...", end="", flush=True)
    last = -1

    def hook(blocks: int, block_size: int, total: int) -> None:
        nonlocal last
        if total <= 0:
            return
        pct = min(100, blocks * block_size * 100 // total)
        if pct == last:            # one repaint per percent, not per block
            return
        last = pct
        print(f"\r  {label}: downloading... {pct}%  ({total / 1e6:.0f} MB)",
              end="", flush=True)

    tmp, _ = urllib.request.urlretrieve(url, reporthook=hook)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(tmp, dest)
    print(f"\r  {label}: ok  ({dest.stat().st_size / 1e6:.1f} MB){' ' * 20}")


def fetch_ytdlp(force: bool) -> None:
    dest = BIN / "yt-dlp.exe"
    if dest.exists() and not force:
        print(f"  yt-dlp: already present ({dest})")
        return
    download(YTDLP_URL, dest, "yt-dlp")


def fetch_ffmpeg(force: bool) -> None:
    have = all((BIN / n).exists() for n in WANTED_FROM_ZIP)
    if have and not force:
        print(f"  ffmpeg: already present ({BIN / 'ffmpeg.exe'})")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / "ffmpeg.zip"
        download(FFMPEG_URL, archive, "ffmpeg")

        print("  ffmpeg: extracting...", end="", flush=True)
        BIN.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                name = member.rsplit("/", 1)[-1]
                if name in WANTED_FROM_ZIP:
                    with zf.open(member) as src, (BIN / name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        print("\r  ffmpeg: ok            ")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-download even if already present (use to update)")
    args = ap.parse_args()

    if sys.platform != "win32":
        sys.exit("bootstrap.py targets Windows; on other platforms install "
                 "yt-dlp and ffmpeg with your package manager.")

    print(f"Installing tools into {BIN}\n")
    fetch_ytdlp(args.force)
    fetch_ffmpeg(args.force)

    missing = [n for n in ("yt-dlp.exe", *WANTED_FROM_ZIP) if not (BIN / n).exists()]
    if missing:
        sys.exit(f"\nStill missing: {', '.join(missing)}")

    print("\nDone. The helper prefers these over anything on PATH.")
    if not (shutil.which("node") or shutil.which("deno")):
        print("\nNOTE: no JavaScript runtime found. YouTube needs one (Node,")
        print("Deno or QuickJS) or some formats will be missing. Install Node")
        print("from https://nodejs.org, or Deno:")
        print("  irm https://deno.land/install.ps1 | iex")


if __name__ == "__main__":
    main()

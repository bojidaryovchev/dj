"""
Locating and running ffmpeg safely.

Two rules, both non-negotiable because this process is handed files by
strangers:

1. Arguments are always a list. ``shell=True`` and f-strings into a command
   line are how a track called ``x.mp3; rm -rf ~`` becomes an incident. Every
   call in this package goes through :func:`run_tool`.
2. stdin is closed. Without that ffmpeg will happily consume the parent's
   stdin when it hits a prompt, which hangs a server and eats a terminal.
   ``-nostdin`` says the same thing to ffmpeg explicitly -- but *only* to
   ffmpeg: ffprobe does not accept the flag and exits with "Option not
   found", hence the switch on :func:`run_tool`.
"""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from ..errors import ToolNotFoundError

__all__ = ["redact_path", "resolve_tool", "run_tool", "tool_version"]


def redact_path(message: str, path: Path) -> str:
    """
    Strip a local filesystem path out of tool output before it is shown.

    ffmpeg quotes the full path it was given in its errors. For an uploaded
    file that path is a server-side temporary name, and echoing it back to
    the caller hands them the temp directory layout and the account the
    service runs as. The basename is all a caller can act on anyway.
    """
    cleaned = message.replace(str(path), path.name)
    parent = str(path.parent)
    return cleaned.replace(parent + "\\", "").replace(parent + "/", "").replace(parent, "")


@lru_cache(maxsize=8)
def resolve_tool(configured: str) -> str:
    """
    Turn a configured name or path into an executable path.

    An absolute path is trusted as given; a bare name is looked up on PATH.
    Raises rather than falling back, because a silent "no ffmpeg" turns into
    a confusing decode error much later.
    """
    if found := shutil.which(configured):
        return found
    raise ToolNotFoundError(
        f"{configured!r} was not found on PATH. Install ffmpeg, or set "
        f"DJTI_FFMPEG_PATH / DJTI_FFPROBE_PATH to its location."
    )


def run_tool(
    executable: str,
    args: list[str],
    *,
    timeout: float,
    capture_stdout: bool = False,
    nostdin: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """
    Run ffmpeg/ffprobe with no shell, no stdin and a hard timeout.

    ``nostdin=False`` for ffprobe, which does not know the flag. Closing the
    child's stdin below covers the same ground either way.
    """
    return subprocess.run(  # noqa: S603 -- list args, shell=False, resolved executable
        [executable, *(["-nostdin"] if nostdin else []), *args],
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        timeout=timeout,
        check=False,
    )


@lru_cache(maxsize=8)
def tool_version(configured: str) -> str | None:
    """First line of ``-version``, for the record of what produced a result."""
    try:
        executable = resolve_tool(configured)
    except ToolNotFoundError:
        return None
    try:
        completed = run_tool(
            executable, ["-version"], timeout=15, capture_stdout=True, nostdin=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    first_line = completed.stdout.decode("utf-8", "replace").splitlines()[:1]
    return first_line[0].strip() if first_line else None

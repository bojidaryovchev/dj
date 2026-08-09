"""
Warp over HTTP.

Two endpoints rather than one, because they return fundamentally different
things and a client should not have to sniff the content type to find out
which it got:

``/plan``
    Always JSON. Analyse, produce the warp map, and say whether rendering is
    warranted. Cheap, and the sensible first call.

``/render``
    Always audio. Refuses with 409 when the plan says the track does not need
    warping, so the expensive, destructive-to-transients operation cannot
    happen by accident — ``force=true`` overrides.

Rendering is synchronous. For a local engine that is the right trade: a job
queue would add a broker, a worker, a result store and a polling protocol to
solve a problem that a two-second wait does not have. The verification summary
travels in response headers so a client gets it without a second request.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import mkdtemp
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from ...config import AnalysisProfile
from ...errors import FileTooLargeError
from ...models import TrackAnalysis
from ...observability import get_logger
from ...warp import warp_track
from ..dependencies import SettingsDep
from ..schemas import WarpPlanResponse
from ..uploads import received_upload

router = APIRouter(prefix="/v1/tracks", tags=["warp"])
log = get_logger(__name__)


def _reject_oversized(request: Request, limit: int) -> None:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise FileTooLargeError(f"upload declares {declared} bytes, over the {limit} byte limit")


@router.post("/warp/plan", response_model=WarpPlanResponse, summary="Plan a grid correction")
async def plan_warp(
    request: Request,
    base_settings: SettingsDep,
    file: Annotated[UploadFile, File(description="Audio file to analyse.")],
    target_bpm: Annotated[float | None, Form(description="Constant tempo to plan against.")] = None,
) -> WarpPlanResponse:
    """Analyse a track and report whether its grid should be corrected."""
    settings = base_settings.with_overrides(profile=AnalysisProfile.WARP)
    _reject_oversized(request, settings.max_upload_bytes)

    async with received_upload(
        file.read, filename=file.filename, max_bytes=settings.max_upload_bytes
    ) as path:
        from ...engine import analyze as run_analysis

        analysis: TrackAnalysis = await run_in_threadpool(
            run_analysis,
            path,
            settings=settings,
            display_name=file.filename or path.name,
            target_bpm=target_bpm,
        )

    return WarpPlanResponse(
        track=analysis.track,
        tempo=analysis.tempo,
        rhythm=analysis.rhythm,
        warp=analysis.warp,
    )


@router.post(
    "/warp/render",
    response_class=FileResponse,
    summary="Render a grid-corrected copy of a track",
    responses={
        200: {"content": {"audio/wav": {}}, "description": "The corrected audio."},
        409: {"description": "Warping is not recommended for this track; pass force=true."},
    },
)
async def render_warp(
    request: Request,
    base_settings: SettingsDep,
    file: Annotated[UploadFile, File(description="Audio file to correct.")],
    target_bpm: Annotated[float | None, Form(description="Constant tempo to warp to.")] = None,
    force: Annotated[
        bool, Form(description="Render even when the grid is already within tolerance.")
    ] = False,
    verify: Annotated[
        bool, Form(description="Re-analyse the output and report the residual error.")
    ] = True,
) -> FileResponse:
    """
    Correct a track's grid and return the audio. Pitch is preserved.

    The source is never modified: the response is a new lossless file.
    """
    settings = base_settings
    _reject_oversized(request, settings.max_upload_bytes)

    workspace = Path(mkdtemp(prefix="djti-warp-"))
    stem = Path(file.filename or "track").stem or "track"
    destination = workspace / f"{stem}.warped.wav"

    try:
        async with received_upload(
            file.read, filename=file.filename, max_bytes=settings.max_upload_bytes
        ) as source:
            outcome = await run_in_threadpool(
                warp_track,
                source,
                output=destination,
                target_bpm=target_bpm,
                force=force,
                verify=verify,
                settings=settings,
            )
    except Exception:
        _cleanup(workspace)
        raise

    if not outcome.rendered:
        _cleanup(workspace)
        raise HTTPException(
            status_code=409,  # Conflict: the request contradicts the analysis
            detail={
                "error": "warp_not_recommended",
                "detail": outcome.skipped_reason,
                "hint": "Send force=true to render anyway.",
            },
        )

    report = outcome.report
    headers = {"X-Warp-Markers": str(report.marker_count if report else 0)}
    if report is not None:
        headers["X-Warp-Target-BPM"] = f"{report.target_bpm:.4f}"
        headers["X-Warp-Stretch-Range"] = (
            f"{report.min_stretch_ratio:.6f}..{report.max_stretch_ratio:.6f}"
        )
        if report.verification is not None:
            headers["X-Warp-Verification"] = json.dumps(
                report.verification.model_dump(mode="json"), separators=(",", ":")
            )

    # The workspace outlives this handler: FileResponse streams from disk, so
    # it can only be removed once the body has actually been sent.
    return FileResponse(
        path=destination,
        media_type="audio/wav",
        filename=destination.name,
        headers=headers,
        background=BackgroundTask(_cleanup, workspace),
    )


def _cleanup(workspace: Path) -> None:
    import shutil

    shutil.rmtree(workspace, ignore_errors=True)

"""
Track analysis over HTTP.

The route does three things: get the bytes onto disk safely, hand them to the
same engine the CLI uses, and serialise the result. It contains no analysis
logic, and it is not coupled to any backend -- swapping Essentia for
something else changes nothing in this file.

Analysis is CPU-bound and blocking, so it runs in a worker thread. Running it
inline would stall the event loop for the whole analysis and stop the service
answering health checks while a track is in flight.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from ...config import AnalysisProfile, Settings
from ...engine import analyze as run_analysis
from ...errors import FileTooLargeError
from ...models import TrackAnalysis
from ...observability import get_logger
from ..dependencies import SettingsDep
from ..uploads import received_upload

router = APIRouter(prefix="/v1/tracks", tags=["tracks"])
log = get_logger(__name__)


def _settings_for(
    base: Settings,
    segments: bool | None,
    max_seconds: float | None,
    profile: AnalysisProfile | None = None,
) -> Settings:
    overrides: dict[str, object] = {}
    if segments is not None:
        overrides["segments_enabled"] = segments
    if max_seconds is not None:
        overrides["max_analysis_seconds"] = max_seconds
    if profile is not None:
        overrides["profile"] = profile
    return base.with_overrides(**overrides) if overrides else base


@router.post(
    "/analyze",
    response_model=TrackAnalysis,
    response_model_exclude_none=False,
    summary="Analyse an uploaded audio file",
)
async def analyze_track(
    request: Request,
    base_settings: SettingsDep,
    file: Annotated[
        UploadFile, File(description="Audio file: mp3, wav, flac, m4a, aac, ogg, opus, aiff.")
    ],
    segments: Annotated[
        bool | None, Form(description="Override time-windowed key analysis.")
    ] = None,
    max_seconds: Annotated[
        float | None, Form(description="Analyse only the first N seconds.")
    ] = None,
    include_beats: Annotated[
        bool, Form(description="Include the beat grid. Thousands of floats on a long track.")
    ] = True,
    profile: Annotated[
        AnalysisProfile | None,
        Form(description="How much of the pipeline to run: basic, full or warp."),
    ] = None,
) -> TrackAnalysis:
    settings = _settings_for(base_settings, segments, max_seconds, profile)

    # Reject on the declared length before reading a byte, when the client
    # was honest enough to declare one. The streaming limit still applies.
    #
    # FileTooLargeError rather than HTTPException so that both routes to a
    # 413 -- this one and the one inside received_upload -- render through
    # the same handler and produce the same body. A client should not have to
    # know which limit caught it to parse the error.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"upload declares {declared} bytes, over the "
            f"{settings.max_upload_bytes} byte limit (DJTI_MAX_UPLOAD_BYTES)"
        )

    async with received_upload(
        file.read, filename=file.filename, max_bytes=settings.max_upload_bytes
    ) as path:
        result = await run_in_threadpool(
            run_analysis, path, settings=settings, display_name=file.filename or path.name
        )

    if not include_beats:
        # Drop both views of the grid, not just the flat one -- the indexed
        # beat list is the larger of the two.
        return result.model_copy(
            update={
                "beats": [],
                "rhythm": result.rhythm.model_copy(update={"beats": []}),
            }
        )
    return result

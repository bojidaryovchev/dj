"""
Liveness and readiness.

``/health`` answers "is this process running?" and must not touch anything
that can be slow or fail -- a health check that shells out to ffmpeg will
take the service down the moment ffmpeg is briefly busy.

``/ready`` answers "can this process actually analyse a track?", which is a
different question and is allowed to be more expensive: it resolves the
external tools and reports which backends are importable. It returns 503 when
the answer is no, so an orchestrator stops sending it work instead of
watching every request fail.
"""

from __future__ import annotations

from fastapi import APIRouter, Response

from ...analysis.registry import available_engines
from ...audio.ffmpeg import resolve_tool, tool_version
from ...errors import ToolNotFoundError
from ..dependencies import SettingsDep
from ..schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthResponse, summary="Liveness")
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse, summary="Readiness")
def ready(response: Response, settings: SettingsDep) -> ReadyResponse:
    engines = available_engines()

    missing: list[str] = []
    versions: dict[str, str | None] = {}
    for label, configured in (("ffmpeg", settings.ffmpeg_path), ("ffprobe", settings.ffprobe_path)):
        try:
            resolve_tool(configured)
        except ToolNotFoundError:
            missing.append(label)
            versions[label] = None
        else:
            versions[label] = tool_version(configured)

    if settings.key_engine.value not in ("auto", *[k for k, v in engines.items() if v]):
        missing.append(f"key engine {settings.key_engine.value}")
    if settings.tempo_engine.value not in ("auto", *[k for k, v in engines.items() if v]):
        missing.append(f"tempo engine {settings.tempo_engine.value}")

    if missing:
        response.status_code = 503  # Service Unavailable

    return ReadyResponse(
        ready=not missing,
        ffmpeg=versions.get("ffmpeg"),
        ffprobe=versions.get("ffprobe"),
        engines=engines,
        key_engine=settings.key_engine.value,
        tempo_engine=settings.tempo_engine.value,
        detail=None if not missing else "unavailable: " + ", ".join(missing),
    )

"""
The HTTP surface.

Exercised through the real app rather than by calling the route functions:
the middleware, the error translation and the multipart handling are most of
what this layer *is*, and calling the handlers directly would skip all three.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dj_intelligence.api.app import REQUEST_ID_HEADER, create_app
from dj_intelligence.api.uploads import safe_suffix
from dj_intelligence.config import Settings
from dj_intelligence.engine import reset_pipeline


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    settings = Settings(key_engine="chroma", tempo_engine="chroma", log_level="ERROR")
    reset_pipeline()
    with TestClient(create_app(settings)) as test_client:
        yield test_client
    reset_pipeline()


# -- ops --------------------------------------------------------------------


def test_health_is_cheap_and_always_answers(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["analysis_version"] == "1.0.0"


def test_ready_reports_the_toolchain(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert "ffmpeg" in body["ffmpeg"].lower()
    assert body["engines"]["chroma"] is True


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    assert client.get("/health").headers[REQUEST_ID_HEADER]


def test_a_supplied_request_id_is_honoured(client: TestClient) -> None:
    """So a trace survives a proxy that already assigned one."""
    response = client.get("/health", headers={REQUEST_ID_HEADER: "trace-me-123"})
    assert response.headers[REQUEST_ID_HEADER] == "trace-me-123"


# -- analysis ---------------------------------------------------------------


@pytest.mark.integration
def test_analyze_returns_the_canonical_document(client: TestClient, f_minor_wav: Path) -> None:
    with f_minor_wav.open("rb") as handle:
        response = client.post(
            "/v1/tracks/analyze", files={"file": ("f_minor.wav", handle, "audio/wav")}
        )
    assert response.status_code == 200
    body = response.json()

    assert body["track"]["filename"] == "f_minor.wav"
    assert len(body["track"]["sha256"]) == 64
    assert body["tonality"]["key"] == "F"
    assert body["tonality"]["mode"] == "minor"
    assert body["dj"]["camelot"] == "4A"
    assert body["tempo"]["bpm"] == pytest.approx(126.0, abs=1.0)
    assert [k["camelot"] for k in body["dj"]["compatible_keys"]] == ["4A", "3A", "5A", "4B"]
    assert body["analysis"]["analysis_version"] == "1.0.0"
    assert body["analysis"]["configuration_fingerprint"]


@pytest.mark.integration
def test_the_uploaded_name_is_reported_not_the_temp_name(
    client: TestClient, f_minor_wav: Path
) -> None:
    with f_minor_wav.open("rb") as handle:
        response = client.post(
            "/v1/tracks/analyze", files={"file": ("Artist - Title.mp3", handle, "audio/mpeg")}
        )
    assert response.json()["track"]["filename"] == "Artist - Title.mp3"


@pytest.mark.integration
def test_beats_can_be_excluded(client: TestClient, f_minor_wav: Path) -> None:
    with f_minor_wav.open("rb") as handle:
        response = client.post(
            "/v1/tracks/analyze",
            files={"file": ("f.wav", handle, "audio/wav")},
            data={"include_beats": "false"},
        )
    body = response.json()
    assert body["beats"] == []
    assert body["tempo"]["beat_count"] > 0  # the count survives; the list does not


@pytest.mark.integration
def test_options_can_be_overridden_per_request(client: TestClient, f_minor_wav: Path) -> None:
    with f_minor_wav.open("rb") as handle:
        response = client.post(
            "/v1/tracks/analyze",
            files={"file": ("f.wav", handle, "audio/wav")},
            data={"segments": "false", "max_seconds": "12"},
        )
    body = response.json()
    assert body["tonal_segments"] == []
    assert body["audio"]["analysed_seconds"] == pytest.approx(12.0, abs=0.2)


# -- failure modes ----------------------------------------------------------


def test_a_corrupt_upload_is_a_415_not_a_500(client: TestClient) -> None:
    response = client.post(
        "/v1/tracks/analyze", files={"file": ("fake.mp3", b"not audio" * 100, "audio/mpeg")}
    )
    assert response.status_code == 415
    body = response.json()
    assert body["error"] == "unsupported_format"
    assert body["request_id"]


def test_errors_do_not_leak_server_paths(client: TestClient) -> None:
    response = client.post(
        "/v1/tracks/analyze", files={"file": ("fake.mp3", b"nope" * 100, "audio/mpeg")}
    )
    detail = response.json()["detail"]
    assert "Temp" not in detail and "tmp" not in detail.lower().replace("djti-", "")


def test_an_empty_upload_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/tracks/analyze", files={"file": ("empty.wav", b"", "audio/wav")})
    assert response.status_code in (415, 422)
    assert response.json()["error"] in ("unsupported_format", "empty_audio", "ingest_failed")


def test_audio_that_is_too_short_is_rejected(client: TestClient, tiny_wav: Path) -> None:
    with tiny_wav.open("rb") as handle:
        response = client.post(
            "/v1/tracks/analyze", files={"file": ("tiny.wav", handle, "audio/wav")}
        )
    assert response.status_code == 422
    assert response.json()["error"] == "audio_too_short"


def test_oversized_uploads_are_refused(f_minor_wav: Path) -> None:
    tiny_limit = Settings(
        key_engine="chroma", tempo_engine="chroma", max_upload_bytes=1024, log_level="ERROR"
    )
    with TestClient(create_app(tiny_limit)) as client, f_minor_wav.open("rb") as handle:
        response = client.post(
            "/v1/tracks/analyze", files={"file": ("big.wav", handle, "audio/wav")}
        )
    assert response.status_code == 413
    assert response.json()["error"] == "file_too_large"


def test_a_missing_file_field_is_a_422(client: TestClient) -> None:
    assert client.post("/v1/tracks/analyze").status_code == 422


# -- upload safety ----------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("track.mp3", ".mp3"),
        ("TRACK.FLAC", ".flac"),
        ("no-extension", ""),
        ("../../../etc/passwd", ""),
        ("x.mp3/../../evil", ""),
        (r"C:\Windows\System32\evil.exe", ".exe"),  # kept, but never used as a path
        ("weird.thisisnotanextension", ""),
        (None, ""),
    ],
)
def test_only_a_plausible_extension_survives(filename: str | None, expected: str) -> None:
    """The uploaded name is never used to build a path -- mkstemp does that --
    so this only has to refuse to carry anything strange across."""
    assert safe_suffix(filename) == expected


@pytest.mark.integration
def test_a_traversal_filename_cannot_escape(client: TestClient, f_minor_wav: Path) -> None:
    with f_minor_wav.open("rb") as handle:
        response = client.post(
            "/v1/tracks/analyze",
            files={"file": ("../../../../evil.wav", handle, "audio/wav")},
        )
    assert response.status_code == 200
    # Echoed back verbatim as a label, and never touched as a path.
    assert response.json()["track"]["filename"] == "../../../../evil.wav"


# -- dj endpoints -----------------------------------------------------------


def test_camelot_neighbours(client: TestClient) -> None:
    response = client.get("/v1/dj/camelot/4A/compatible")
    assert response.status_code == 200
    assert [(k["camelot"], k["relationship"]) for k in response.json()] == [
        ("4A", "same_key"),
        ("3A", "adjacent_minus"),
        ("5A", "adjacent_plus"),
        ("4B", "relative_major"),
    ]


def test_extended_neighbours(client: TestClient) -> None:
    response = client.get("/v1/dj/camelot/4A/compatible", params={"extended": True})
    assert len(response.json()) == 7


def test_a_bad_camelot_key_is_a_400(client: TestClient) -> None:
    assert client.get("/v1/dj/camelot/99Z/compatible").status_code == 400


def test_compatibility_scoring(client: TestClient) -> None:
    response = client.post(
        "/v1/dj/compatibility",
        json={"track_a": {"camelot": "4A", "bpm": 126}, "track_b": {"camelot": "5A", "bpm": 126.5}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["score"] > 0.9
    assert body["harmonic_relationship"] == "adjacent_plus"
    assert body["components"]["harmonic"] == pytest.approx(0.95)
    assert body["rules_version"]


def test_compatibility_with_partial_data(client: TestClient) -> None:
    response = client.post(
        "/v1/dj/compatibility",
        json={"track_a": {"bpm": 126}, "track_b": {"camelot": "5A", "bpm": 126}},
    )
    assert response.status_code == 200
    assert response.json()["components"]["harmonic"] is None


def test_a_bad_compatibility_payload_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/dj/compatibility",
        json={"track_a": {"camelot": "not-a-key", "bpm": 126}, "track_b": {"camelot": "5A"}},
    )
    assert response.status_code == 400


def test_unknown_fields_are_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/dj/compatibility",
        json={"track_a": {"camelot": "4A", "bpmm": 126}, "track_b": {"camelot": "5A"}},
    )
    assert response.status_code == 422


def test_openapi_is_served(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/v1/tracks/analyze" in schema["paths"]
    assert "/v1/dj/compatibility" in schema["paths"]

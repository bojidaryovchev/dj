"""
Request-scoped dependencies.

Routes take their configuration from the application they are mounted in, not
from the process-wide cache. That distinction is not academic: an app built
with ``create_app(Settings(max_upload_bytes=1024))`` whose routes read the
global settings would advertise a limit it does not enforce, which is worse
than having no limit at all. It also makes the API testable at more than one
configuration in a single process.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from ..config import Settings, get_settings

__all__ = ["SettingsDep", "settings_from_app"]


def settings_from_app(request: Request) -> Settings:
    """The settings this app was created with, falling back to the global."""
    configured = getattr(request.app.state, "settings", None)
    return configured if isinstance(configured, Settings) else get_settings()


SettingsDep = Annotated[Settings, Depends(settings_from_app)]

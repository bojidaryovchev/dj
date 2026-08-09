# DJ Track Intelligence.
#
# The container is where Essentia actually runs. It publishes manylinux and
# macOS wheels only -- there is no Windows wheel and no plausible way to build
# one -- so a Windows host falls back to the portable chroma/librosa backend
# while this image gets the reference implementation. That difference is the
# main reason the image exists, ahead of any deployment concern.
#
#   docker build -t dj-track-intelligence .
#   docker run --rm -p 8000:8000 dj-track-intelligence
#   docker run --rm -v "$PWD/music:/music:ro" dj-track-intelligence dj-analyze /music/track.mp3

# 3.12 rather than 3.13+: Essentia's newest wheels do not cover every
# interpreter, and the analysis stack is the constraint here, not the runtime.
FROM python:3.12-slim-bookworm AS base

# ffmpeg is not optional -- it is the decoder for every format we accept.
# libgomp1 is required by Essentia's OpenMP-linked extension module.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- dependencies ---------------------------------------------------------
# Installed from pyproject before the source is copied, so that editing a
# module does not reinstall numba and scipy. The dependency list is read out
# of pyproject rather than duplicated here, because a second copy of it would
# drift. A one-line `python -c` rather than a heredoc: heredocs in RUN need a
# BuildKit syntax directive, and this image should build under any builder.
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir hatchling \
    && python -c "import tomllib; p = tomllib.load(open('pyproject.toml','rb'))['project']; print('\n'.join(p['dependencies'] + p['optional-dependencies']['essentia']))" > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# --- application ----------------------------------------------------------
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir --no-deps -e .

# Fail the build rather than the first request if the image cannot analyse.
RUN ffmpeg -version > /dev/null \
    && python -c "import essentia.standard as es; es.KeyExtractor(); es.RhythmExtractor2013(method='multifeature'); print('essentia ok')" \
    && python -c "import librosa; print('librosa', librosa.__version__)"

# Time stretching is only usable if ffmpeg was built against librubberband, and
# "the filter is listed" is not the same as "the filter works". Stretch two
# seconds of tone to four and check the output really is about four seconds, so
# a base image that quietly drops librubberband fails here rather than at the
# first warp request.
#
# "About" is not slackness: a phase vocoder returns whole analysis frames, so
# this comes out at 3.95 s rather than 4.00. That inexactness is why the
# renderer pins every segment to its target length instead of trusting the
# stretcher to land on it.
RUN ffmpeg -v error -f lavfi -i "sine=frequency=440:duration=2" \
        -af "rubberband=tempo=0.5:pitch=1" -y /tmp/stretch-check.wav \
    && python -c "import subprocess, sys; \
out = subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0','/tmp/stretch-check.wav'], capture_output=True, text=True).stdout; \
d = float(out); print(f'rubberband stretched 2.00s -> {d:.2f}s'); \
sys.exit(0 if 3.7 < d < 4.3 else 1)" \
    && rm /tmp/stretch-check.wav

# Essentia is present here, so prefer it. `auto` would too, but being
# explicit means a broken Essentia install fails loudly instead of quietly
# falling back to a different engine than the image promises.
ENV DJTI_KEY_ENGINE=essentia \
    DJTI_TEMPO_ENGINE=essentia \
    DJTI_LOG_FORMAT=json \
    DJTI_HOST=0.0.0.0 \
    DJTI_PORT=8000

# Analysis is CPU-bound; nothing here needs root.
RUN useradd --create-home --uid 10001 analyst \
    && chown -R analyst:analyst /app
USER analyst

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# No --log-config: create_app() installs this package's structured logging
# and takes uvicorn's own handlers over, so both emit the same JSON.
CMD ["uvicorn", "dj_intelligence.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]

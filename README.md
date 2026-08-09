<img src="logo.png" alt="DJ" width="120" align="right">

# dj

Tools for building and understanding a DJ crate.

Two halves, and they meet at a folder of MP3s:

| | |
|---|---|
| **[packages/extension](packages/extension/)** + **[packages/crate-helper](packages/crate-helper/)** | **DJ Crate** — collect every open YouTube / SoundCloud tab, untick what you don't want, one button, tagged MP3s in your crate folder. A Chrome MV3 extension talking to a local `yt-dlp` server. |
| **[packages/track-intelligence](packages/track-intelligence/)** | **Track Intelligence** — an audio analysis engine. Key, Camelot, BPM, beat grid, tonal structure and loudness, over a CLI and an HTTP API. |

They are independent today: getting tracks and understanding tracks are
separate problems, and neither needs the other to be useful. The obvious join
— analyse each download as it lands and write the key and BPM into the ID3
tags — is on the roadmap and not built.

---

## Getting started

**Download tracks** → [packages/crate-helper/README.md](packages/crate-helper/README.md)

```powershell
packages\crate-helper\start.cmd
```

Double-click it. It installs anything missing, creates a config with a fresh
token, fetches `yt-dlp` and `ffmpeg`, and starts the server. Then load
`packages\extension` unpacked in Chrome and paste the token into its options
page.

**Analyse tracks** → [packages/track-intelligence/README.md](packages/track-intelligence/README.md)

```bash
cd packages/track-intelligence
uv sync --extra dev
uv run dj-analyze /path/to/track.mp3
```

```text
Key          F minor
Camelot      4A
Tempo        126.04 BPM
```

Or over HTTP:

```bash
uv run dj-analyze serve
curl -X POST -F "file=@track.mp3" http://localhost:8000/v1/tracks/analyze
```

---

## Layout

```text
packages/
├── extension/           Chrome MV3 extension — tab list, selection, cookie export
├── crate-helper/        local Python server — yt-dlp job queue, downloads
└── track-intelligence/  audio analysis engine — key, Camelot, BPM, structure
```

Three deployable units, two languages, no shared build. Each package owns its
own dependencies and its own README; there is no root-level toolchain to
install, because there is nothing for one to do. The extension has no build
step, the crate helper is standard-library-only Python, and the analysis
engine is a `uv` project.

> **Moved.** The extension and helper used to sit at the repository root as
> `extension/` and `helper/`. They are now under `packages/`. Point Chrome's
> **Load unpacked** at `packages\extension` — the extension ID is pinned by the
> `key` field in its manifest, so it stays the same and the helper's origin
> check keeps working.

---

## Legal

Downloading from YouTube is against its Terms of Service, and tracks played at
a paid gig normally need to come from a licensed source. Your call — noted once
so it's on the record.

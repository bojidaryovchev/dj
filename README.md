# DJ Crate

Collect every open YouTube / SoundCloud tab, untick the ones you don't want,
hit one button, and get tagged MP3s in your crate folder.

Two pieces:

- **`extension/`** — Chrome MV3 extension. Lists supported tabs, owns the UI,
  and exports browser cookies.
- **`helper/`** — local Python server that actually runs `yt-dlp`. Extensions
  are sandboxed and can't spawn processes, so this bridge is unavoidable.

They talk over `http://127.0.0.1:8765`, gated by a shared token *and* an
origin check pinned to one extension ID.

---

## Install

### 1. Start the helper

Double-click **`helper\start.cmd`**. That's the whole install step — it takes
the machine from nothing to running:

| | |
|---|---|
| Python missing? | installs it via `winget` |
| No Node or Deno? | installs Node LTS via `winget` |
| No `config.json`? | creates one and prints a fresh token |
| No `yt-dlp` / `ffmpeg`? | runs `bootstrap.py` to fetch them into `bin/` |
| all present | falls straight through and starts the server |

Every step is idempotent, so it's safe to run every time — on a normal day it
goes straight to the server. Pass `-Yes` to install prerequisites without
prompting.

Leave the window open while you download.

Prefer to do it by hand? `python bootstrap.py` then `python server.py`.

**What gets fetched.** `yt-dlp.exe`, `ffmpeg.exe` and `ffprobe.exe` land in
`helper/bin/` (~310 MB, one time). The helper prefers `bin/` over anything on
PATH, which pins you to yt-dlp's own patched ffmpeg build instead of whatever
old copy happens to be first on PATH. Update both later with:

```powershell
python bootstrap.py --force
```

**Why a JavaScript runtime.** YouTube gates playback behind JS challenges
yt-dlp has to execute. Without one you get
`WARNING: No supported JavaScript runtime could be found` and **some formats
silently go missing** — a worse rip with no error to tell you. Deno is
yt-dlp's default, but Node works too and `config.json` passes
`--js-runtimes node` for that reason. Supported: Deno, Node, QuickJS. Bun is
deprecated.

> `winget` ships with Windows 11. Chocolatey would need admin rights *and*
> installing Chocolatey itself first, so it isn't used here.

### 2. Load the extension

1. `chrome://extensions` → enable **Developer mode**
2. **Load unpacked** → select `d:\repos\dj\extension`
3. Confirm the ID reads `kbfggmpcibfbdaeclojnppjkcccphcbo`

The ID is pinned by the `key` field in the manifest. Don't remove it — the
helper's origin check is tied to it, and without it Chrome reassigns the ID on
every reload.

### 3. Pair them

Open the extension's **options** page and paste the `token` — `start.cmd`
prints it the first time it creates `config.json`, and it's in that file
after. Set your destination folder. Save.

---

## Use

1. Open tabs as you dig.
2. Click the toolbar icon. Every supported tab appears, all ticked.
3. Untick what you don't want.
4. **Download selected.**

Three at a time by default, with live progress per row. Close the popup
whenever you like — the helper owns the queue, so reopening re-attaches to
whatever is still running.

**Sync cookies** exports your YouTube and SoundCloud cookies to the helper.
Do this when you hit *"Sign in to confirm you're not a bot"*, which is the
usual YouTube failure now. See the note below on why this button exists.

---

## Things worth knowing

### Cookies, and why the extension exports them itself

`yt-dlp --cookies-from-browser chrome` **does not work on Windows** any more.
Chrome 127+ binds the cookie encryption key to the Chrome binary
(app-bound encryption), so no external tool can read the jar.

The extension holds the `cookies` permission, so it reads them via
`chrome.cookies.getAll()` — httpOnly included — writes a Netscape
`cookies.txt`, and posts it to the helper. That sidesteps the problem
completely, and means you don't need a third-party cookie exporter. Several of
the popular ones have shipped as malware, so this is the safer path anyway.

Use a throwaway Google account for this. A main account driving a downloader
can get flagged.

### Playlists

A tab sitting on a Mix or a playlist has its `&list=` stripped, so you get the
one track you were listening to. If you actually want the whole playlist, the
row shows a **+ whole playlist** link.

### Expect some failures

SoundCloud's extractor is maintained but not perfect — DRM-flagged and Go+
tracks 404 at the format step and there is no workaround. Failures are
reported per row rather than killing the batch.

### Parallelism

Three workers. Above four you meaningfully raise your odds of a 429 from
YouTube. Change `concurrency` in `config.json` and restart the helper.

### Audio

`bestaudio[ext=m4a]/bestaudio` → MP3 V0 (~245 kbps measured), with embedded
artwork and `Artist - Title` split into proper ID3 fields. Rekordbox, Serato
and Traktor all read this; none of them read the Opus/WebM that YouTube
actually serves, which is why the transcode is there.

`helper/archive.txt` records what you've pulled, so re-scanning your tabs won't
re-download. Delete it to start fresh.

---

## Layout

```
extension/
  manifest.json     MV3 + pinned key
  popup.html/js/css tab list, selection, progress
  options.html/js   connection + download settings
  lib.js            tab classification, API client, cookie export
helper/
  server.py         job queue + yt-dlp pool
  start.cmd         double-click launcher (thin wrapper)
  start.ps1         installs prerequisites, then runs the server
  bootstrap.py      fetches yt-dlp + ffmpeg into bin/
  config.json       token, destination, concurrency   (secret — keep local)
  bin/              yt-dlp.exe, ffmpeg.exe, ffprobe.exe  (not committed)
  cookies/          exported jars                     (secret — keep local)
  archive.txt       already-downloaded track ids
```

Tool resolution order is `bin/` → `ytdlp_path` in config → PATH.

## Legal

Downloading from YouTube is against its Terms of Service, and tracks played at
a paid gig normally need to come from a licensed source. Your call — noted once
so it's on the record.

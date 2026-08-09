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

### 1. Get the tools

Only **Python 3** and a **JavaScript runtime** have to be on the machine
already. Everything else the bootstrap fetches:

```powershell
cd d:\repos\dj\helper
python bootstrap.py
```

That pulls `yt-dlp.exe` and `ffmpeg.exe` / `ffprobe.exe` into `helper/bin/`
(~310 MB, one time). The helper prefers `bin/` over anything on PATH, so this
works on a machine with neither installed — and it pins you to yt-dlp's own
patched ffmpeg build rather than whatever old copy is first on PATH.

Re-run with `--force` to update both later:

```powershell
python bootstrap.py --force
```

**About the JavaScript runtime.** YouTube gates playback behind JS challenges
yt-dlp has to execute. Without a runtime you get
`WARNING: No supported JavaScript runtime could be found` and **some formats
silently go missing** — a worse rip with no error to tell you.

Deno is yt-dlp's default, but **Node also works and you already have v24**, so
there's nothing to install here. `config.json` passes `--js-runtimes node` for
that reason. Supported: Deno, Node, QuickJS. Bun is deprecated.

If you ever move this to a machine with neither, install one:

```powershell
irm https://deno.land/install.ps1 | iex     # or just install Node
```

### 2. Start the helper

```powershell
cd d:\repos\dj\helper
python server.py
```

Or double-click `helper\start.cmd`. It prints its config and flags anything
missing. Leave it running while you download.

### 3. Load the extension

1. `chrome://extensions` → enable **Developer mode**
2. **Load unpacked** → select `d:\repos\dj\extension`
3. Confirm the ID reads `kbfggmpcibfbdaeclojnppjkcccphcbo`

The ID is pinned by the `key` field in the manifest. Don't remove it — the
helper's origin check is tied to it, and without it Chrome reassigns the ID on
every reload.

### 4. Pair them

Open the extension's **options** page and paste the `token` from
`helper/config.json`. Set your destination folder. Save.

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

`helper/library.json` records where each download landed, so re-scanning your
tabs won't re-download. Rows you already hold come back **unticked** with the
filename, before you queue anything.

The check is against the filesystem, not a list of ids: delete or move a track
and it becomes downloadable again on the next scan. That's why this replaced
yt-dlp's `--download-archive`, which stores only `youtube <id>` and so goes on
insisting you have files you deleted months ago. Delete `library.json` to start
fresh. (`archive.txt` is left over from that scheme and is no longer read.)

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
  bootstrap.py      fetches yt-dlp + ffmpeg into bin/
  start.cmd         double-click launcher
  config.json       token, destination, concurrency   (secret — keep local)
  bin/              yt-dlp.exe, ffmpeg.exe, ffprobe.exe  (not committed)
  cookies/          exported jars                     (secret — keep local)
  library.json      url -> where the file landed
```

Tool resolution order is `bin/` → `ytdlp_path` in config → PATH.

## Legal

Downloading from YouTube is against its Terms of Service, and tracks played at
a paid gig normally need to come from a licensed source. Your call — noted once
so it's on the record.

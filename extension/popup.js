import { classifyTab, dedupe, api, exportCookies, getSettings } from "./lib.js";

const el = {
  status: document.getElementById("status"),
  list: document.getElementById("list"),
  empty: document.getElementById("empty"),
  count: document.getElementById("count"),
  toggleAll: document.getElementById("toggleAll"),
  refresh: document.getElementById("refresh"),
  cookies: document.getElementById("cookies"),
  download: document.getElementById("download"),
  dest: document.getElementById("dest"),
};

let items = [];   // classified tabs
let poll = null;
let pollFails = 0;

// --- helper connection ---------------------------------------------------

async function checkHelper() {
  const { token } = await getSettings();
  if (!token) {
    setStatus("no token — open options", "bad");
    return false;
  }
  try {
    const h = await api("/health");
    el.dest.textContent = h.download_dir;
    const warn = [];
    if (!h.ytdlp) warn.push("yt-dlp missing");
    if (!h.ffmpeg) warn.push("ffmpeg missing");
    if (!h.js_runtime) warn.push("no JS runtime");
    if (!h.cookies.length) warn.push("no cookies synced");
    setStatus(warn.length ? warn.join(" · ") : `helper ok · ${h.concurrency} workers`,
              warn.length ? "bad" : "good");
    return true;
  } catch (e) {
    setStatus(`helper unreachable (${e.message})`, "bad");
    return false;
  }
}

function setStatus(text, cls = "") {
  el.status.textContent = text;
  el.status.className = `status ${cls}`;
}

// --- tab scanning --------------------------------------------------------

async function scanTabs() {
  const tabs = await chrome.tabs.query({});
  items = dedupe(tabs.map(classifyTab).filter(Boolean));
  render();
  await markHeld();
}

/**
 * Untick anything the helper still has on disk, so you can see what's
 * genuinely new before queueing rather than after.
 */
async function markHeld() {
  let have;
  try {
    ({ have } = await api("/library"));
  } catch {
    return;   // not paired yet, or helper down -- leave everything ticked
  }

  for (const row of rows()) {
    const name = have[row.dataset.url];
    if (!name) continue;
    row.querySelector("input").checked = false;
    const state = row.querySelector(".state");
    state.className = "state skipped";
    state.textContent = `↷ already have ${name}`;
  }
  updateCount();
}

function render() {
  el.list.replaceChildren();
  el.empty.classList.toggle("hidden", items.length > 0);

  for (const it of items) {
    el.list.appendChild(renderRow(it));
  }
  updateCount();
}

function renderRow(it) {
  const li = document.createElement("li");
  li.className = "row";
  li.dataset.url = it.url;

  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = true;
  cb.addEventListener("change", updateCount);

  const meta = document.createElement("div");
  meta.className = "meta";

  const name = document.createElement("div");
  name.className = "name";
  name.textContent = it.title;
  name.title = it.url;

  const sub = document.createElement("div");
  sub.className = "sub";

  const tag = document.createElement("span");
  tag.className = `tag ${it.site === "youtube" ? "yt" : "sc"}`;
  tag.textContent = it.site === "youtube" ? "YT" : "SC";
  sub.append(tag);

  if (it.kind === "playlist") {
    const pl = document.createElement("span");
    pl.className = "tag pl";
    pl.textContent = "playlist";
    sub.append(pl);
  }

  const state = document.createElement("span");
  state.className = "state";
  state.textContent = "";
  sub.append(state);

  // A tab sitting on a Mix: offer the playlist explicitly rather than
  // silently queueing 200 tracks.
  if (it.playlistId) {
    const link = document.createElement("a");
    link.href = "#";
    link.textContent = "+ whole playlist";
    link.style.color = "#66b9e0";
    link.addEventListener("click", (ev) => {
      ev.preventDefault();
      items.push({
        ...it,
        url: `https://www.youtube.com/playlist?list=${it.playlistId}`,
        title: `${it.title} — full playlist`,
        kind: "playlist",
        playlistId: null,
      });
      render();
    });
    sub.append(link);
  }

  const bar = document.createElement("div");
  bar.className = "progress";
  bar.append(document.createElement("i"));

  meta.append(name, sub, bar);
  li.append(cb, meta);
  return li;
}

function rows() {
  return [...el.list.querySelectorAll(".row")];
}

function selected() {
  return rows().filter((r) => r.querySelector("input").checked);
}

function updateCount() {
  const n = selected().length;
  el.count.textContent = `${n} of ${items.length} selected`;
  el.download.disabled = n === 0;
}

// --- download + progress -------------------------------------------------

async function startDownload() {
  const chosen = selected().map((r) => {
    const it = items.find((i) => i.url === r.dataset.url);
    return { url: it.url, title: it.title };
  });
  if (!chosen.length) return;

  el.download.disabled = true;
  try {
    await api("/jobs", { method: "POST", body: { items: chosen } });
    startPolling();
  } catch (e) {
    setStatus(`submit failed: ${e.message}`, "bad");
    el.download.disabled = false;
  }
}

function startPolling() {
  stopPolling();
  pollFails = 0;
  poll = setInterval(refreshJobs, 700);
  refreshJobs();
}

function stopPolling() {
  if (poll) clearInterval(poll);
  poll = null;
}

async function refreshJobs() {
  let jobs;
  try {
    ({ jobs } = await api("/jobs"));
    pollFails = 0;
  } catch (e) {
    // Give up rather than retrying a rejection twice a second forever — that
    // buries the actual cause under hundreds of identical log lines.
    if (++pollFails >= 3) {
      stopPolling();
      setStatus(`progress unavailable: ${e.message}`, "bad");
      el.download.disabled = selected().length === 0;
    }
    return;
  }

  // Match on URL, not job id, so reopening the popup re-attaches to jobs
  // that are still running from a previous session.
  const byUrl = new Map(jobs.map((j) => [j.url, j]));
  let active = 0;

  for (const row of rows()) {
    const job = byUrl.get(row.dataset.url);
    if (!job) continue;

    const state = row.querySelector(".state");
    const bar = row.querySelector(".progress > i");

    if (job.status === "running") {
      state.className = "state";
      state.textContent = job.message
        ? `${job.message}…`
        : `${job.percent.toFixed(0)}%  ${job.speed}  ETA ${job.eta}`;
      bar.style.width = `${job.percent}%`;
      active++;
    } else if (job.status === "queued") {
      state.className = "state";
      state.textContent = "queued";
      active++;
    } else {
      state.className = `state ${job.status}`;
      state.textContent = job.status === "done" ? `✓ ${job.message}`
                        : job.status === "skipped" ? `↷ ${job.message}`
                        : `✕ ${job.message}`;
      bar.style.width = job.status === "done" ? "100%" : "0";
    }
  }

  if (active === 0) {
    stopPolling();
    el.download.disabled = selected().length === 0;
  }
}

// --- wiring --------------------------------------------------------------

el.toggleAll.addEventListener("change", () => {
  for (const r of rows()) r.querySelector("input").checked = el.toggleAll.checked;
  updateCount();
});

el.refresh.addEventListener("click", async () => {
  await api("/clear", { method: "POST" }).catch(() => {});
  await scanTabs();
});

el.cookies.addEventListener("click", async () => {
  el.cookies.disabled = true;
  try {
    const counts = await exportCookies();
    setStatus(`cookies: yt ${counts.youtube}, sc ${counts.soundcloud}`, "good");
  } catch (e) {
    setStatus(`cookie sync failed: ${e.message}`, "bad");
  } finally {
    el.cookies.disabled = false;
  }
});

el.download.addEventListener("click", startDownload);

(async function init() {
  await scanTabs();
  const ok = await checkHelper();
  if (ok) startPolling();   // pick up jobs still running from a previous popup
})();

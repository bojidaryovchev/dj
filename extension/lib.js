// Shared helpers: tab classification, helper-server client, cookie export.

const SC_RESERVED = new Set([
  "discover", "stream", "you", "search", "upload", "settings", "pages",
  "charts", "feed", "notifications", "messages", "tags", "people", "terms",
  "imprint", "pro", "premium", "mobile", "jobs",
]);

/**
 * Decide whether a tab is downloadable and normalise its URL.
 * Returns null for anything we don't support.
 */
export function classifyTab(tab) {
  if (!tab.url) return null;

  let u;
  try {
    u = new URL(tab.url);
  } catch {
    return null;
  }

  const host = u.hostname.replace(/^www\./, "");
  const title = cleanTitle(tab.title || tab.url);

  // --- YouTube ---------------------------------------------------------
  if (host === "youtu.be") {
    const id = u.pathname.slice(1);
    if (!id) return null;
    return item(tab, `https://www.youtube.com/watch?v=${id}`, title, "youtube", "track");
  }

  if (host === "youtube.com" || host === "music.youtube.com" || host === "m.youtube.com") {
    const v = u.searchParams.get("v");
    if (v) {
      // Deliberately drop &list= -- otherwise a tab sitting on a Mix would
      // queue 200 tracks you never asked for. Playlists are opt-in below.
      const hadList = u.searchParams.has("list");
      return item(tab, `https://www.youtube.com/watch?v=${v}`, title, "youtube",
                  "track", hadList ? u.searchParams.get("list") : null);
    }
    if (u.pathname.startsWith("/shorts/")) {
      const id = u.pathname.split("/")[2];
      if (id) return item(tab, `https://www.youtube.com/watch?v=${id}`, title, "youtube", "track");
    }
    if (u.pathname === "/playlist" && u.searchParams.get("list")) {
      return item(tab, `https://www.youtube.com/playlist?list=${u.searchParams.get("list")}`,
                  title, "youtube", "playlist");
    }
    return null;
  }

  // --- SoundCloud ------------------------------------------------------
  if (host === "soundcloud.com" || host === "m.soundcloud.com") {
    const parts = u.pathname.split("/").filter(Boolean);
    if (parts.length < 2) return null;
    if (SC_RESERVED.has(parts[0])) return null;

    const kind = parts[1] === "sets" ? "playlist" : "track";
    if (kind === "track" && parts.length !== 2) return null;

    return item(tab, `https://soundcloud.com${u.pathname}`, title, "soundcloud", kind);
  }

  return null;
}

function item(tab, url, title, site, kind, playlistId = null) {
  return { tabId: tab.id, url, title, site, kind, playlistId, windowId: tab.windowId };
}

/** Strip the noise browsers and uploaders bolt onto titles. */
function cleanTitle(raw) {
  return raw
    .replace(/^\(\d+\)\s*/, "")                       // unread-count prefix
    .replace(/\s*[-–|]\s*YouTube( Music)?$/i, "")
    .replace(/\s*\|\s*Free Listening on SoundCloud$/i, "")
    .trim();
}

/** Drop duplicate URLs, keeping the first tab that had each. */
export function dedupe(items) {
  const seen = new Set();
  return items.filter((it) => {
    if (seen.has(it.url)) return false;
    seen.add(it.url);
    return true;
  });
}

// --- helper server client ----------------------------------------------

export async function getSettings() {
  const d = await chrome.storage.local.get({
    serverUrl: "http://127.0.0.1:8765",
    token: "",
  });
  return d;
}

export async function api(path, { method = "GET", body } = {}) {
  const { serverUrl, token } = await getSettings();
  const res = await fetch(`${serverUrl}${path}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      "X-Auth-Token": token,
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).error || detail;
    } catch { /* non-JSON error body */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

// --- cookies ------------------------------------------------------------

const COOKIE_DOMAINS = {
  youtube: [".youtube.com", ".google.com"],
  soundcloud: [".soundcloud.com"],
};

/**
 * Export cookies in Netscape format.
 *
 * This exists because Chrome 127+ app-bound encryption on Windows makes
 * yt-dlp's --cookies-from-browser unusable. We have the cookies permission,
 * so we can read them (httpOnly included) and hand yt-dlp a jar directly.
 */
export async function exportCookies() {
  const jars = {};

  for (const [name, domains] of Object.entries(COOKIE_DOMAINS)) {
    const lines = ["# Netscape HTTP Cookie File", "# Exported by DJ Crate"];
    const seen = new Set();

    for (const domain of domains) {
      const cookies = await chrome.cookies.getAll({ domain });
      for (const c of cookies) {
        const key = `${c.domain}\t${c.path}\t${c.name}`;
        if (seen.has(key)) continue;
        seen.add(key);

        const includeSub = c.domain.startsWith(".") ? "TRUE" : "FALSE";
        const secure = c.secure ? "TRUE" : "FALSE";
        const expiry = Math.floor(c.expirationDate || 0);
        lines.push([c.domain, includeSub, c.path, secure, expiry, c.name, c.value].join("\t"));
      }
    }
    jars[name] = lines.join("\n") + "\n";
  }

  await api("/cookies", { method: "POST", body: { jars } });
  return Object.fromEntries(
    Object.entries(jars).map(([k, v]) => [k, v.split("\n").length - 3])
  );
}

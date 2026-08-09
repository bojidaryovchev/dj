import { api } from "./lib.js";

const f = (id) => document.getElementById(id);
const msg = f("msg");

async function load() {
  const stored = await chrome.storage.local.get({
    serverUrl: "http://127.0.0.1:8765",
    token: "",
  });
  f("serverUrl").value = stored.serverUrl;
  f("token").value = stored.token;

  // Download settings are owned by the helper, so read them from there.
  try {
    const h = await api("/health");
    f("downloadDir").value = h.download_dir;
    f("concurrency").value = h.concurrency;
    f("audioFormat").value = h.audio_format;
    say("connected to helper", "good");
  } catch (e) {
    say(`helper unreachable — start it, then reload (${e.message})`, "bad");
  }

  const local = await chrome.storage.local.get({ useCookies: true, useArchive: true });
  f("useCookies").checked = local.useCookies;
  f("useArchive").checked = local.useArchive;
}

function say(text, cls = "") {
  msg.textContent = text;
  msg.className = cls === "good" ? "state done" : cls === "bad" ? "state error" : "";
}

f("save").addEventListener("click", async () => {
  await chrome.storage.local.set({
    serverUrl: f("serverUrl").value.trim().replace(/\/$/, ""),
    token: f("token").value.trim(),
    useCookies: f("useCookies").checked,
    useArchive: f("useArchive").checked,
  });

  try {
    await api("/config", {
      method: "POST",
      body: {
        download_dir: f("downloadDir").value.trim(),
        concurrency: Number(f("concurrency").value),
        audio_format: f("audioFormat").value,
        use_cookies: f("useCookies").checked,
        use_archive: f("useArchive").checked,
      },
    });
    say("saved", "good");
  } catch (e) {
    say(`saved locally, but helper rejected the update: ${e.message}`, "bad");
  }
});

load();

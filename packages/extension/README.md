# DJ Crate — Chrome extension

The UI half of DJ Crate. Lists every supported tab, owns the selection and
progress display, and exports browser cookies to the helper.

It cannot download anything by itself — extensions are sandboxed and cannot
spawn processes — so it talks to the local server in
[`../crate-helper`](../crate-helper/) over `http://127.0.0.1:8765`.

**Install and usage are documented in
[../crate-helper/README.md](../crate-helper/README.md).** Short version:
`chrome://extensions` → Developer mode → **Load unpacked** → select this
folder, then paste the helper's token into the options page.

```text
manifest.json     MV3 + the pinned `key` that fixes the extension ID
popup.html/js/css tab list, selection, per-row progress
options.html/js   connection + download settings
lib.js            tab classification, helper client, cookie export
icons/            generated from ../../logo.png — see icons/README.md
```

Do not remove the `key` field from the manifest. It pins the extension ID, the
helper's origin check is tied to that ID, and without it Chrome reassigns a new
one on every reload.

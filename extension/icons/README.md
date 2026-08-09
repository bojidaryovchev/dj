# Icons

Generated from `logo.png` at the repo root. To regenerate after changing it:

```powershell
cd d:\repos\dj
foreach ($s in 16,32,48,128) {
  helper\bin\ffmpeg.exe -y -i logo.png `
    -vf "crop=880:880:167:160,scale=${s}:${s}:flags=lanczos" `
    extension\icons\icon$s.png
}
```

The crop trims the dead black border (content sits at `830x784+192+208`,
found with `cropdetect`) and squares it up around the artwork centre, so the
mark fills the tile instead of floating in padding. Lanczos keeps the
letterforms crisp on the way down.

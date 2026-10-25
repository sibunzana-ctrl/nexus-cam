# NexusCam

Live background replacement, virtual webcam and recorder for Linux, built for trading content under **Nexus Trading Systems**.

![NexusCam](docs/screenshot.png)

NexusCam takes your real webcam, cuts you out of the frame, and puts a new background behind you: a blur, an image, a looping video, a solid colour, or a **live window such as a TradingView chart**. The result goes to a virtual webcam (`NexusCam`) that Zoom, Meet, OBS, Discord and browsers can use, and can be recorded straight to MP4.

## Features

- **Backgrounds:** None, Blur, Image, Video, Colour and Screen (share any window or monitor through the desktop's own picker)
- **Presenter controls:** shrink yourself and move left or right so the chart stays visible
- **Phone as camera**, two ways: **Wi-Fi** (scan a QR code; Android and iPhone) or **USB cable** (Android; works even when the router blocks devices from seeing each other). Nothing to install on the phone
- **Virtual webcam** via v4l2loopback, so any app can use it
- **Recording** to MP4 with microphone and/or computer audio (videos, TradingView alerts), mixed into one track; encodes on an NVIDIA GPU (NVENC) when available, otherwise on the CPU (x264)
- **Mic picker with a live level meter**, plus an automatic check that warns you if a recording's audio is silent
- **1080p output** even from a 720p webcam, so charts stay sharp
- **Fast:** about 8–13 ms per 1080p frame for compositing
- Hotkeys: `1`–`6` background, `Space` go live, `R` record

## Requirements

- Pop!_OS 22.04+ / Ubuntu 22.04+ (tested on Pop!_OS 24.04 with COSMIC, Wayland)
- A webcam
- Optional: an NVIDIA GPU for hardware video encoding

## Install

```bash
git clone https://github.com/sibunzana-ctrl/nexus-cam.git
cd nexus-cam
bash setup.sh
```

`setup.sh` installs the virtual camera driver, screen-capture and audio tools, sets up a Python environment, and adds **NexusCam** to your app menu. If Secure Boot is on, you'll be asked to enroll a key on the next reboot so the virtual camera driver can load.

## Use

1. Open **NexusCam** from the app menu (or run `./run.sh`).
2. **Setup** tab: choose your webcam and resolution.
3. **Scene** tab: pick a background. For a chart, choose **Screen** and select the TradingView window.
4. **Go live**, then select **NexusCam** as the camera in your call or streaming app.
5. **Record** saves to `~/Videos/NexusCam/`.

## Phone camera

### Wi-Fi

1. **Setup** tab → Camera → **Phone camera (Wi-Fi)**. A QR code appears in the preview.
2. Scan it with your phone (phone and PC on the same Wi-Fi).
3. The browser shows a security warning, because NexusCam uses its own certificate. Tap **Advanced → Proceed**. This only happens once per phone.
4. Allow camera access and tap **Start**, then press **Start camera** in NexusCam.

Use **Front/Back** and **720p/1080p** on the phone page. Hold the phone sideways for a landscape picture. The link contains a private key, so other devices on your network can't send video to NexusCam.

### USB cable (Android)

Use this if the Wi-Fi page won't load. Some routers have "AP isolation", which stops devices on the same Wi-Fi from reaching each other.

1. One-time setup on the phone: **Settings → About phone → Software information → tap Build number 7 times**, then **Settings → Developer options → turn on USB debugging**.
2. **Setup** tab → Camera → **Phone camera (USB cable)**.
3. Plug the phone in and tap **Allow** on the "Allow USB debugging?" prompt (tick *Always allow*).
4. Press **Open on phone** in NexusCam. The page opens in your phone's browser with no security warning. Tap **Start**.

## Troubleshooting

| Problem | Fix |
|---|---|
| Background shows instead of you | Cut-out tab → **Invert mask** |
| Halo or flicker around your head | Raise **Cut-out tightness** and **Flicker smoothing**; add light on your face |
| Webcam stuck at 15 fps | More light on your face (NexusCam already disables the camera's low-light frame-rate drop) |
| Silent recording | Audio tab → pick another input and press **Test**; use the one whose meter moves |
| Video plays but no sound in the player | `sudo apt install gstreamer1.0-libav` (setup.sh does this) — the player was missing its AAC decoder |
| Computer audio heard twice / echo | Use headphones, so the mic doesn't also pick up your speakers |
| Screen share errors | See `~/.nexuscam/screen.log` |
| Recording errors | See `~/.nexuscam/ffmpeg.log` |
| Phone page won't load (Wi-Fi) | Use **Phone camera (USB cable)** instead. Otherwise, phone and PC must be on the same Wi-Fi; some routers' "guest" networks block devices from seeing each other. If you use a firewall: `sudo ufw allow 8765/tcp` |

Settings are stored in `~/.nexuscam/settings.json`.

---

© Nexus Trading Systems

#!/usr/bin/env bash
# NexusCam installer for Pop!_OS / Ubuntu  (v6 — phone camera)
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Installing system packages (virtual camera, Tk, screen capture)"
sudo apt update
sudo apt install -y v4l2loopback-dkms v4l2loopback-utils python3-venv python3-tk \
  python3-gi gir1.2-gstreamer-1.0 gstreamer1.0-pipewire \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good xdg-desktop-portal ffmpeg v4l-utils pulseaudio-utils zenity gstreamer1.0-libav openssl

echo "==> Loading the NexusCam virtual camera at /dev/video10"
if [ ! -e /dev/video10 ]; then
  sudo modprobe -r v4l2loopback 2>/dev/null || true
  sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="NexusCam" exclusive_caps=1
fi

echo "==> Making it load automatically on boot"
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf >/dev/null
echo 'options v4l2loopback devices=1 video_nr=10 card_label="NexusCam" exclusive_caps=1' \
  | sudo tee /etc/modprobe.d/v4l2loopback.conf >/dev/null

echo "==> Creating Python environment (system Python, so screen capture can use GStreamer)"
# Use the system Python explicitly — not conda's — because python3-gi is built for it.
# Re-use the environment if it is already the right kind; rebuild it otherwise.
if ! grep -q "include-system-site-packages = true" "$DIR/venv/pyvenv.cfg" 2>/dev/null; then
  rm -rf "$DIR/venv"
  /usr/bin/python3 -m venv --system-site-packages "$DIR/venv"
fi
"$DIR/venv/bin/pip" install --upgrade pip
"$DIR/venv/bin/pip" install mediapipe opencv-python pyvirtualcam pillow numpy mss qrcode

echo "==> Creating launcher"
cat > "$DIR/run.sh" <<EOF
#!/usr/bin/env bash
exec "$DIR/venv/bin/python" "$DIR/nexuscam.py" "\$@"
EOF
chmod +x "$DIR/run.sh"

mkdir -p "$HOME/.local/share/applications"
cat > "$HOME/.local/share/applications/nexuscam.desktop" <<EOF
[Desktop Entry]
Name=NexusCam
Comment=Live background replacement virtual webcam
Exec=$DIR/run.sh
Icon=camera-web
Terminal=false
Type=Application
Categories=AudioVideo;Video;
EOF

echo ""
echo "Done. Launch NexusCam from your app menu, or run: $DIR/run.sh"

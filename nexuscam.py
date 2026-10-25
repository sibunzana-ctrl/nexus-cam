#!/usr/bin/env python3
"""
NexusCam — live background replacement virtual webcam for Linux (Pop!_OS).

Real webcam -> person segmentation (MediaPipe) -> new background -> virtual
webcam (v4l2loopback via pyvirtualcam). In Zoom / Meet / Teams / OBS /
Discord / the browser, pick "NexusCam" as your camera.

Background modes: None, Blur, Image, Video, Colour, Screen (a live window or
monitor — e.g. TradingView or your engine dashboard — captured through the
desktop's screen-share picker on Wayland, or directly on X11).

Hotkeys:  1 None | 2 Blur | 3 Image | 4 Video | 5 Colour | 6 Screen
          Space = Go live / stop
"""

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.request

import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp

try:
    import pyvirtualcam
except ImportError:  # app still runs as a preview without it
    pyvirtualcam = None

# ----------------------------------------------------------------- constants
APP_DIR = os.path.join(os.path.expanduser("~"), ".nexuscam")
SETTINGS_FILE = os.path.join(APP_DIR, "settings.json")
MODEL_PATH = os.path.join(APP_DIR, "selfie_segmenter.tflite")
SCREEN_LOG = os.path.join(APP_DIR, "screen.log")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_segmenter/float16/latest/selfie_segmenter.tflite")

MODES = ["none", "blur", "image", "video", "colour", "screen"]
MODE_LABELS = {"none": "None", "blur": "Blur", "image": "Image",
               "video": "Video", "colour": "Colour", "screen": "Screen"}
RESOLUTIONS = ["640x480", "1280x720", "1920x1080"]
OUTPUT_SIZES = ["Same as camera", "1280x720", "1920x1080"]
SEG_W = 640  # width the mask is computed at; upscaled once at the end
POSITIONS = ["Left", "Centre", "Right"]
PREVIEW_W = 640
VIRTUAL_NAMES = ("nexuscam", "dummy", "obs", "loopback", "virtual")

DEFAULTS = {
    "camera": "", "resolution": "1280x720", "fps": 30,
    "mode": "blur", "blur": 45, "softness": 5, "threshold": 50,
    "smoothing": 50, "mirror": True, "invert": False,
    "image_path": "", "video_path": "", "colour": "#0b1d2a",
    "vcam_device": "/dev/video10",
    "screen_fit": True, "screen_token": "",
    "presenter_pos": "Centre", "presenter_size": 100,
    "output_size": "Same as camera",
    "record_audio": True, "mic_source": "",
    "record_dir": os.path.join(os.path.expanduser("~"), "Videos", "NexusCam"),
}

# Theme
BG, PANEL, FG, MUTED, ACCENT, LIVE = (
    "#0f141a", "#171e26", "#e6edf3", "#8b98a5", "#2f81f7", "#e5484d")


# ----------------------------------------------------------------- helpers
def list_cameras():
    """Return [(device_path, name)] for primary capture nodes."""
    base = "/sys/class/video4linux"
    cams = []
    if not os.path.isdir(base):
        return cams

    def num(d):
        digits = "".join(c for c in d if c.isdigit())
        return int(digits) if digits else 0

    for dev in sorted(os.listdir(base), key=num):
        path = os.path.join(base, dev)
        try:
            with open(os.path.join(path, "index")) as f:
                if f.read().strip() not in ("", "0"):
                    continue  # metadata node, not a real stream
        except OSError:
            pass
        try:
            with open(os.path.join(path, "name")) as f:
                name = f.read().strip()
        except OSError:
            name = "Unknown camera"
        cams.append((f"/dev/{dev}", name))
    return cams


def is_virtual(name, path, vcam_device):
    return path == vcam_device or any(v in name.lower() for v in VIRTUAL_NAMES)


def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def cover(img, w, h):
    """Resize + centre-crop img to exactly fill w x h (object-fit: cover)."""
    ih, iw = img.shape[:2]
    if (iw, ih) == (w, h):
        return img
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(round(iw * s))), max(h, int(round(ih * s)))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x, y = (nw - w) // 2, (nh - h) // 2
    return img[y:y + h, x:x + w]


def contain(img, w, h, fill=(18, 12, 6)):
    """Resize img to fit entirely inside w x h, letterboxed (object-fit: contain)."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, min(w, int(round(iw * s)))), max(1, min(h, int(round(ih * s))))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.full((h, w, 3), fill, np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y:y + nh, x:x + nw] = small
    return out


def default_background(w, h, message=None):
    """Branded fallback backdrop: navy gradient, faint grid, Nexus wordmark."""
    top = np.array([46, 26, 12], np.float32)
    bot = np.array([18, 10, 4], np.float32)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = (top * (1 - t) + bot * t) * np.ones((1, w, 1), np.float32)
    img = img.astype(np.uint8)
    step = max(40, w // 24)
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (58, 38, 22), 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (58, 38, 22), 1)
    scale = w / 1280
    font = cv2.FONT_HERSHEY_DUPLEX
    text = "NEXUS TRADING SYSTEMS"
    (tw, _), _ = cv2.getTextSize(text, font, 1.0 * scale, 2)
    cv2.putText(img, text, (w - tw - int(40 * scale), h - int(40 * scale)),
                font, 1.0 * scale, (247, 129, 47), max(1, int(2 * scale)),
                cv2.LINE_AA)
    if message:
        (mw, mh), _ = cv2.getTextSize(message, font, 0.8 * scale, 1)
        cv2.putText(img, message, ((w - mw) // 2, int(60 * scale) + mh),
                    font, 0.8 * scale, (200, 190, 180), max(1, int(scale)),
                    cv2.LINE_AA)
    return img


def is_wayland():
    return (os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
            or bool(os.environ.get("WAYLAND_DISPLAY")))


# ----------------------------------------------------------------- screen capture
class ScreenSource:
    """Keeps the latest frame of a chosen monitor or window on a background
    thread. Wayland (COSMIC, GNOME, KDE…): xdg-desktop-portal ScreenCast +
    PipeWire via GStreamer, so the desktop's own picker chooses the source.
    X11: grabs the primary monitor with mss."""

    PORTAL = "org.freedesktop.portal.Desktop"
    PORTAL_PATH = "/org/freedesktop/portal/desktop"
    SCREENCAST = "org.freedesktop.portal.ScreenCast"

    def __init__(self, restore_token=""):
        self._latest = None
        self.seq = 0  # bumps on every new frame, so consumers can cache work
        self.error = None
        self.status = "Not sharing"
        self.restore_token = restore_token
        self._running = False
        self._thread = None

    @property
    def latest(self):
        return self._latest

    @latest.setter
    def latest(self, frame):
        self._latest = frame
        self.seq += 1

    @property
    def active(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, reuse=True):
        self.stop()
        self.latest = None
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(reuse,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._thread = None
        self.latest = None
        self.status = "Not sharing"

    def _run(self, reuse):
        try:
            if is_wayland():
                self._run_portal(reuse)
            else:
                self._run_x11()
        except Exception as e:
            if self._running:
                self.error = f"Screen capture stopped:\n{e}"
        finally:
            self._running = False
            if self.status not in ("Cancelled",):
                self.status = "Not sharing"

    # -- X11
    def _run_x11(self):
        import mss
        with mss.mss() as sct:
            mon = sct.monitors[1]
            self.status = "Sharing primary monitor"
            while self._running:
                shot = np.asarray(sct.grab(mon))[:, :, :3]
                self.latest = np.ascontiguousarray(shot)
                time.sleep(1 / 15)

    # -- Wayland portal
    def _run_portal(self, reuse):
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import GLib, Gio, Gst
        except Exception:
            raise RuntimeError("PyGObject / GStreamer not found.\n"
                               "Re-run setup.sh to install screen-capture support.")
        Gst.init(None)
        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        sender = bus.get_unique_name()[1:].replace(".", "_")
        counter = [0]

        def request(method, signature, args, options, timeout=180):
            """Call a portal method and wait for its Request::Response."""
            counter[0] += 1
            token = f"nexuscam_{os.getpid()}_{counter[0]}"
            options = dict(options)
            options["handle_token"] = GLib.Variant("s", token)
            req_path = f"/org/freedesktop/portal/desktop/request/{sender}/{token}"
            loop = GLib.MainLoop()
            out = {}
            started = time.time()

            def on_response(*a):
                code, results = a[5].unpack()
                out["code"], out["results"] = code, results
                loop.quit()

            def watchdog():
                if not self._running or time.time() - started > timeout:
                    out.setdefault("code", 1)
                    out.setdefault("results", {})
                    loop.quit()
                    return False
                return True

            sid = bus.signal_subscribe(self.PORTAL, "org.freedesktop.portal.Request",
                                       "Response", req_path, None,
                                       Gio.DBusSignalFlags.NONE, on_response)
            try:
                bus.call_sync(self.PORTAL, self.PORTAL_PATH, self.SCREENCAST, method,
                              GLib.Variant(signature, (*args, options)),
                              GLib.VariantType("(o)"), Gio.DBusCallFlags.NONE, -1, None)
                GLib.timeout_add(200, watchdog)
                loop.run()
            finally:
                bus.signal_unsubscribe(sid)
            return out.get("code", 2), out.get("results", {})

        # which source types does this desktop offer? (1 = monitor, 2 = window)
        types = 1
        try:
            v = bus.call_sync(self.PORTAL, self.PORTAL_PATH, "org.freedesktop.DBus.Properties",
                              "Get", GLib.Variant("(ss)", (self.SCREENCAST, "AvailableSourceTypes")),
                              GLib.VariantType("(v)"), Gio.DBusCallFlags.NONE, -1, None)
            types = (int(v.unpack()[0]) & 3) or 1
        except Exception:
            pass

        session = None
        pipeline = None
        try:
            self.status = "Opening screen-share…"
            code, res = request("CreateSession", "(a{sv})", (),
                                {"session_handle_token": GLib.Variant("s", f"nexuscam_s{os.getpid()}")})
            if code != 0:
                raise RuntimeError("The desktop refused to start a screen-share session.")
            session = res["session_handle"]

            opts = {"types": GLib.Variant("u", types),
                    "multiple": GLib.Variant("b", False),
                    "persist_mode": GLib.Variant("u", 2)}
            if reuse and self.restore_token:
                opts["restore_token"] = GLib.Variant("s", self.restore_token)
            code, _ = request("SelectSources", "(oa{sv})", (session,), opts)
            if code != 0:
                self.status = "Cancelled"
                return

            self.status = "Pick a screen or window in the dialog…"
            code, res = request("Start", "(osa{sv})", (session, ""), {})
            if code != 0:
                self.status = "Cancelled"
                return
            streams = res.get("streams") or []
            if not streams:
                raise RuntimeError("No screen or window was shared.")
            node_id = int(streams[0][0])
            if res.get("restore_token"):
                self.restore_token = res["restore_token"]

            def open_remote():
                result, fds = bus.call_with_unix_fd_list_sync(
                    self.PORTAL, self.PORTAL_PATH, self.SCREENCAST, "OpenPipeWireRemote",
                    GLib.Variant("(oa{sv})", (session, {})), GLib.VariantType("(h)"),
                    Gio.DBusCallFlags.NONE, -1, None, None)
                return fds.get(result.unpack()[0])

            # The portal session stays open; if the GStreamer stream breaks
            # (PipeWire 1.0 drops it on window resizes / format changes), we
            # rebuild just the pipeline on a fresh connection — no new picker.
            # Only give up after repeated failures in a short window.
            failures = []
            while self._running:
                fd = open_remote()
                try:
                    self._stream(Gst, fd, node_id)
                    break  # returned normally: we were asked to stop
                except _ShareEnded:
                    raise RuntimeError("The screen share was ended from the desktop.")
                except RuntimeError as e:
                    now = time.time()
                    failures = [t for t in failures if now - t < 30] + [now]
                    self._log(f"stream error (attempt {len(failures)}): {e}")
                    if len(failures) >= 5:
                        raise RuntimeError(f"{str(e).splitlines()[0]}\n\nThe stream failed 5 times in "
                                           f"30 seconds. Details: {SCREEN_LOG}")
                    self.status = "Reconnecting…"
                    time.sleep(0.5)
                finally:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
        finally:
            if session is not None:
                try:
                    bus.call_sync(self.PORTAL, session, "org.freedesktop.portal.Session",
                                  "Close", None, None, Gio.DBusCallFlags.NONE, -1, None)
                except Exception:
                    pass

    def _stream(self, Gst, fd, node_id):
        """Run one GStreamer pipeline until stopped (returns) or it fails (raises)."""
        pipeline = Gst.parse_launch(
            f"pipewiresrc name=src fd={fd} path={node_id} do-timestamp=true ! "
            "queue max-size-buffers=2 leaky=downstream ! "
            "videoconvert ! video/x-raw,format=BGRx ! "
            "appsink name=sink max-buffers=1 drop=true sync=false")
        src = pipeline.get_by_name("src")
        # copy frames out of the compositor's buffers (avoids DMA-BUF mapping
        # failures) and re-send the last frame when the window is idle
        for prop, val in (("always-copy", True), ("resend-last", True), ("keepalive-time", 1000)):
            if src.find_property(prop) is not None:
                try:
                    src.set_property(prop, val)
                except Exception:
                    pass
        sink = pipeline.get_by_name("sink")
        gbus = pipeline.get_bus()
        try:
            pipeline.set_state(Gst.State.PLAYING)
            self.status = "Sharing"
            while self._running:
                msg = gbus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if msg is not None:
                    if msg.type == Gst.MessageType.ERROR:
                        err, dbg = msg.parse_error()
                        raise RuntimeError(f"{err.message}\n{dbg or ''}".strip())
                    raise _ShareEnded()
                sample = sink.emit("try-pull-sample", 200 * Gst.MSECOND)
                if sample is None:
                    continue  # nothing changed on screen; keep last frame
                st = sample.get_caps().get_structure(0)
                w, h = st.get_value("width"), st.get_value("height")
                buf = sample.get_buffer()
                ok, info = buf.map(Gst.MapFlags.READ)
                if not ok:
                    continue
                try:
                    stride = info.size // h
                    arr = np.frombuffer(info.data, np.uint8, count=stride * h).reshape(h, stride)
                    self.latest = arr[:, :w * 4].reshape(h, w, 4)[:, :, :3].copy()
                finally:
                    buf.unmap(info)
        finally:
            pipeline.set_state(Gst.State.NULL)

    @staticmethod
    def _log(text):
        try:
            os.makedirs(APP_DIR, exist_ok=True)
            with open(SCREEN_LOG, "a") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text.replace("\n", " | ") + "\n")
        except OSError:
            pass


class _ShareEnded(Exception):
    """The desktop (or user, via the system indicator) ended the share."""


# ----------------------------------------------------------------- camera grabber
class FrameGrabber(threading.Thread):
    """Reads (and MJPG-decodes) camera frames on its own thread, so decoding
    the next frame overlaps with processing the current one."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.frame = None
        self.seq = 0
        self.running = True
        self._cond = threading.Condition()

    def run(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self.frame = f
                self.seq += 1
                self._cond.notify_all()

    def get(self, last_seq, timeout=1.0):
        """Wait for a frame newer than last_seq; returns (seq, frame)."""
        with self._cond:
            self._cond.wait_for(lambda: self.seq != last_seq or not self.running, timeout)
            return self.seq, self.frame

    def stop(self):
        self.running = False
        with self._cond:
            self._cond.notify_all()
        self.join(timeout=2)


# ----------------------------------------------------------------- recording
_NVENC = None


def nvenc_available():
    """True if ffmpeg can encode H.264 on the NVIDIA GPU (checked once)."""
    global _NVENC
    if _NVENC is None:
        try:
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                 "-i", "color=c=black:s=320x240:d=0.2", "-c:v", "h264_nvenc",
                 "-f", "null", "-"], capture_output=True, timeout=20)
            _NVENC = r.returncode == 0
        except Exception:
            _NVENC = False
    return _NVENC


def list_mic_sources():
    """[(pulse_source_name, label)] — real inputs first, then 'computer audio'
    monitors. Works on PipeWire (pipewire-pulse) and PulseAudio."""
    if not shutil.which("pactl"):
        return []
    env = dict(os.environ, LC_ALL="C")
    try:
        out = subprocess.run(["pactl", "list", "sources"], capture_output=True,
                             text=True, timeout=5, env=env).stdout
        default = subprocess.run(["pactl", "get-default-source"], capture_output=True,
                                 text=True, timeout=5, env=env).stdout.strip()
    except Exception:
        return []
    return parse_pactl_sources(out, default)


def parse_pactl_sources(out, default=""):
    mics, monitors = [], []
    name = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Description:") and name:
            desc = line.split(":", 1)[1].strip()
            if name.endswith(".monitor"):
                monitors.append((name, "Computer audio: " + desc.replace("Monitor of ", "")))
            else:
                mics.append((name, desc + ("  (system default)" if name == default else "")))
            name = None
    return mics + monitors


class MicMeter:
    """Reads a few ms of audio at a time from one input and keeps its peak
    level in dBFS, so you can see whether the mic actually hears you."""

    def __init__(self, source):
        self.level = -90.0
        self.error = None
        self.proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "pulse",
             "-i", source or "default", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        while True:
            data = self.proc.stdout.read(1600)  # 50 ms
            if not data:
                break
            pcm = np.frombuffer(data[: len(data) // 2 * 2], np.int16)
            peak = int(np.abs(pcm.astype(np.int32)).max()) if pcm.size else 0
            db = 20 * np.log10(max(peak, 1) / 32768.0)
            self.level = max(db, self.level - 3.0)  # fast attack, slow fall
        try:
            rc = self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            rc = None
        if rc not in (None, 0, -15, 255):
            err = (self.proc.stderr.read() or b"").decode(errors="ignore").strip()
            self.error = err.splitlines()[-1] if err else "The microphone could not be opened."

    def stop(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


def audio_peak_db(path):
    """Loudest point of the file's audio track in dBFS (None if no audio)."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn",
                            "-af", "volumedetect", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=120)
    except Exception:
        return None
    m = re.search(r"max_volume:\s*(-?[\d.]+|-inf) dB", r.stderr)
    if not m:
        return None
    return -91.0 if m.group(1) == "-inf" else float(m.group(1))


def mp4_is_playable(path):
    """True if ffprobe can read a duration from the file."""
    if not shutil.which("ffprobe"):
        return os.path.getsize(path) > 0
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", path],
                           capture_output=True, text=True, timeout=15)
        return r.returncode == 0 and float(r.stdout.strip() or 0) > 0
    except Exception:
        return False


class Recorder:
    """Pipes NexusCam's output frames (+ microphone) into ffmpeg -> MP4.
    Frames are queued and written on their own thread so a slow disk or
    encoder never stalls the camera; if the queue fills, frames are dropped."""

    MIC_INPUT = ["-f", "pulse", "-i", "default"]

    def __init__(self, w, h, fps, path, audio=True, audio_input=None, source=""):
        self.path = path
        self.audio = audio
        self.warning = None
        if audio and audio_input is None:
            audio_input = ["-f", "pulse", "-i", source or "default"]
        self.started = time.time()
        self.dropped = 0
        self.error = None
        self.log_path = os.path.join(APP_DIR, "ffmpeg.log")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-thread_queue_size", "512",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-use_wallclock_as_timestamps", "1", "-i", "-"]
        if audio:
            cmd += ["-thread_queue_size", "1024"] + (audio_input or self.MIC_INPUT)
        cmd += ["-map", "0:v"]
        if audio:
            cmd += ["-map", "1:a", "-c:a", "aac", "-b:a", "160k", "-af", "aresample=async=1"]
        if nvenc_available():  # RTX GPU does the encoding; CPU stays free for the cut-out
            self.encoder = "GPU (NVENC)"
            cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                    "-cq", "21", "-b:v", "0"]
        else:
            self.encoder = "CPU (x264)"
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
        if audio:
            cmd += ["-shortest"]  # trim the mic tail to where the video ends
        cmd += ["-pix_fmt", "yuv420p", "-fps_mode", "cfr", "-r", str(fps),
                "-movflags", "+faststart", path]
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=self._log)
        self._q = queue.Queue(maxsize=8)
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    @property
    def alive(self):
        return self.proc.poll() is None

    @property
    def elapsed(self):
        return time.time() - self.started

    def write(self, frame_bgr):
        try:
            self._q.put_nowait(np.ascontiguousarray(frame_bgr))
        except queue.Full:
            self.dropped += 1

    def _writer(self):
        while True:
            f = self._q.get()
            if f is None:
                break
            try:
                self.proc.stdin.write(f.tobytes())
            except (BrokenPipeError, OSError, ValueError):
                break

    def _log_tail(self):
        try:
            with open(self.log_path) as f:
                return "".join(f.readlines()[-8:]).strip()
        except OSError:
            return ""

    def stop(self):
        """Finish the file. Returns an error message, or None on success."""
        try:
            self._q.put(None, timeout=2)
        except queue.Full:
            pass
        self._thread.join(timeout=5)
        try:
            self.proc.stdin.close()  # end of video
        except Exception:
            pass
        # The microphone never "ends", so ffmpeg won't stop by itself just
        # because the video did. Ask it to finish the way Ctrl+C does: it then
        # writes the MP4 index (moov) and exits cleanly.
        try:
            self.proc.wait(timeout=0.7)  # let ffmpeg drain the last frames
        except subprocess.TimeoutExpired:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
                self._log.close()
                return ("Recording failed: ffmpeg did not finish the file in time, "
                        "so it is probably unplayable.\n\n" + self._log_tail())
        self._log.close()
        if self.proc.returncode not in (0, 255) or not os.path.exists(self.path):
            return "Recording failed.\n\n" + (self._log_tail() or "ffmpeg exited unexpectedly.")
        if not mp4_is_playable(self.path):
            return "Recording finished but the file is not playable.\n\n" + self._log_tail()
        if self.audio:
            peak = audio_peak_db(self.path)
            if peak is not None and peak < -50:
                self.warning = (f"The video saved, but its sound is silent (loudest point "
                                f"{peak:.0f} dB).\n\nThe selected microphone isn't hearing you. "
                                "Pick another input under Record and press 'Test' — "
                                "use the one whose meter moves when you speak.")
        return None


# ----------------------------------------------------------------- segmentation
class Segmenter:
    """Person mask (float 0..1). Uses MediaPipe legacy API, or the Tasks API on
    newer MediaPipe builds that dropped mp.solutions."""

    def __init__(self):
        self.kind = None
        try:
            self._legacy = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1)
            self.kind = "legacy"
            return
        except Exception:
            pass
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        os.makedirs(APP_DIR, exist_ok=True)
        if not os.path.exists(MODEL_PATH):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        opts = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False)
        self._tasks = vision.ImageSegmenter.create_from_options(opts)
        self.kind = "tasks"

    def mask(self, rgb):
        if self.kind == "legacy":
            return self._legacy.process(rgb).segmentation_mask
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=np.ascontiguousarray(rgb))
        res = self._tasks.segment(img)
        m = res.confidence_masks[-1].numpy_view()
        return np.squeeze(m).astype(np.float32)

    def close(self):
        try:
            (self._legacy if self.kind == "legacy" else self._tasks).close()
        except Exception:
            pass


# ----------------------------------------------------------------- engine
class Engine(threading.Thread):
    """Capture -> segment -> composite -> virtual cam, on a background thread."""

    def __init__(self, settings, live=False, screen=None):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._settings = dict(settings)
        self.screen = screen
        self.recorder = None
        self.running = False
        self.live = live
        self.vcam_open = False
        self.frame_out = None
        self.size = (0, 0)
        self.fps = 0.0
        self.proc_ms = 0.0
        self.error = None
        self._screen_key, self._screen_bg = None, None
        self._img_key, self._img = None, None
        self._video, self._video_path = None, None
        self._placeholder = {}
        self._prev_mask = None

    # settings are swapped atomically; the loop reads a snapshot each frame
    def update(self, s):
        with self._lock:
            self._settings = dict(s)

    def snapshot(self):
        with self._lock:
            return dict(self._settings)

    def stop(self):
        self.running = False

    def run(self):
        self.running = True
        s = self.snapshot()
        w, h = map(int, s["resolution"].split("x"))
        fps = int(s["fps"])

        cap = cv2.VideoCapture(s["camera"], cv2.CAP_V4L2)
        if not cap.isOpened():
            self.error = (f"Could not open camera {s['camera']}.\n"
                          "Close any other app using it and try again.")
            self.running = False
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock_framerate(s["camera"])

        ok, frame = cap.read()
        if not ok:
            self.error = f"Camera {s['camera']} opened but returned no frames."
            cap.release()
            self.running = False
            return

        # output size: the camera's own, or a fixed size (e.g. 1080p output from a
        # 720p webcam — the screen/chart stays sharp, only you are upscaled)
        h, w = frame.shape[:2]
        out_size = None
        if s.get("output_size", "Same as camera") != "Same as camera":
            ow, oh = map(int, s["output_size"].split("x"))
            if (ow, oh) != (w, h):
                out_size = (ow, oh)
                w, h = ow, oh
        self.size = (w, h)

        try:
            seg = Segmenter()
        except Exception as e:
            self.error = f"Could not load the segmentation model:\n{e}"
            cap.release()
            self.running = False
            return

        grabber = FrameGrabber(cap)
        grabber.start()
        cam = None
        seq = 0
        frames, t0 = 0, time.time()
        try:
            while self.running:
                seq, frame = grabber.get(seq)
                if frame is None:
                    continue
                t_start = time.perf_counter()
                s = self.snapshot()
                if out_size is not None:
                    frame = cover(frame, *out_size)
                if s["mirror"]:
                    frame = cv2.flip(frame, 1)
                out = self.process(frame, s, seg)
                rec = self.recorder
                if rec is not None:
                    rec.write(out)
                rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                dt = (time.perf_counter() - t_start) * 1000
                self.proc_ms = dt if self.proc_ms == 0 else 0.9 * self.proc_ms + 0.1 * dt

                if self.live and cam is None:
                    cam = self._open_vcam(s, w, h, fps)
                elif not self.live and cam is not None:
                    cam.close()
                    cam = None
                if cam is not None:
                    try:
                        cam.send(rgb)
                    except Exception as e:
                        self.error = f"Virtual camera stopped: {e}"
                        cam.close()
                        cam, self.live = None, False
                self.vcam_open = cam is not None

                self.frame_out = rgb
                frames += 1
                now = time.time()
                if now - t0 >= 1.0:
                    self.fps = frames / (now - t0)
                    frames, t0 = 0, now
        finally:
            grabber.stop()
            cap.release()
            if cam is not None:
                cam.close()
            self.vcam_open = False
            seg.close()
            if self._video is not None:
                self._video.release()

    @staticmethod
    def _lock_framerate(dev):
        """Most webcams drop to 15 fps in dim light to lengthen exposure.
        Turning off 'dynamic framerate' keeps them at 30 (image gets a bit
        darker/noisier instead)."""
        if not shutil.which("v4l2-ctl"):
            return
        for ctrl in ("exposure_dynamic_framerate=0", "exposure_auto_priority=0"):
            try:
                subprocess.run(["v4l2-ctl", "-d", dev, "-c", ctrl],
                               capture_output=True, timeout=3)
            except Exception:
                pass

    def _open_vcam(self, s, w, h, fps):
        dev = s["vcam_device"]
        if pyvirtualcam is None:
            self.error = "pyvirtualcam is not installed (run setup.sh)."
            self.live = False
            return None
        if not os.path.exists(dev):
            self.error = (f"{dev} does not exist — the virtual camera driver "
                          "isn't loaded.\nRun setup.sh, or:\n"
                          "sudo modprobe v4l2loopback devices=1 video_nr=10 "
                          "card_label=NexusCam exclusive_caps=1")
            self.live = False
            return None
        try:
            return pyvirtualcam.Camera(width=w, height=h, fps=fps, device=dev,
                                       fmt=pyvirtualcam.PixelFormat.RGB)
        except Exception as e:
            self.error = f"Could not start the virtual camera on {dev}:\n{e}"
            self.live = False
            return None

    # -- compositing
    def process(self, frame, s, seg):
        """Mask work happens at SEG_W (cheap); full resolution is touched only
        by one mask upscale and one SIMD blend."""
        if s["mode"] == "none":
            self._prev_mask = None
            return frame
        h, w = frame.shape[:2]
        sw = min(w, SEG_W)
        sh = max(1, int(round(h * sw / w)))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA) if sw < w else frame
        m = seg.mask(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if m.shape[:2] != (sh, sw):
            m = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_LINEAR)
        m = m.astype(np.float32, copy=False)
        if s["invert"]:
            m = 1.0 - m

        # temporal smoothing kills edge flicker
        a = s["smoothing"] / 100.0 * 0.85
        if self._prev_mask is not None and self._prev_mask.shape == m.shape:
            m = cv2.addWeighted(self._prev_mask, a, m, 1.0 - a, 0.0)
        self._prev_mask = m

        # threshold ramp + feathered edge (softness is in output pixels)
        thr = s["threshold"] / 100.0
        m = np.clip((m - (thr - 0.1)) * 5.0, 0.0, 1.0)
        f = int(round(int(s["softness"]) * sw / w))
        if f > 0:
            m = cv2.GaussianBlur(m, (f * 2 + 1, f * 2 + 1), 0)

        bg = self.background(frame, s)

        # presenter size / position (not for Blur — the blurred room would ghost)
        person = frame
        if s["mode"] != "blur":
            person, m = self.place_presenter(frame, m, s)

        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.blendLinear(person, bg, m, 1.0 - m)

    @staticmethod
    def presenter_matrix(w, h, s):
        scale = max(0.3, min(1.0, s["presenter_size"] / 100.0))
        side = {"Left": -1, "Centre": 0, "Right": 1}.get(s["presenter_pos"], 0)
        if scale >= 0.999 and side == 0:
            return None
        dx = side * w * (0.5 - 0.45 * scale)
        tx = w / 2 + dx - scale * w / 2
        ty = h - scale * h
        return np.float32([[scale, 0, tx], [0, scale, ty]])

    @classmethod
    def place_presenter(cls, frame, m, s):
        """Scale the presenter about the bottom edge and slide left/right.
        The frame is warped at full res; the mask at its own (small) res."""
        h, w = frame.shape[:2]
        M = cls.presenter_matrix(w, h, s)
        if M is None:
            return frame, m
        person = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mh, mw = m.shape[:2]
        Ms = M.copy()
        Ms[0, 2] *= mw / w
        Ms[1, 2] *= mh / h
        m = cv2.warpAffine(m, Ms, (mw, mh), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return person, m

    def placeholder(self, w, h, message):
        key = (w, h, message)
        if key not in self._placeholder:
            self._placeholder = {key: default_background(w, h, message)}
        return self._placeholder[key]

    def background(self, frame, s):
        h, w = frame.shape[:2]
        mode = s["mode"]

        if mode == "blur":
            k = max(3, (int(s["blur"]) // 4) | 1)
            small = cv2.resize(frame, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (k, k), 0)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        if mode == "colour":
            return np.full(frame.shape, hex_to_bgr(s["colour"]), np.uint8)

        if mode == "image":
            key = (s["image_path"], w, h)
            if key != self._img_key:
                img = cv2.imread(s["image_path"]) if s["image_path"] else None
                self._img = cover(img, w, h) if img is not None else default_background(w, h)
                self._img_key = key
            return self._img

        if mode == "video":
            path = s["video_path"]
            if path != self._video_path:
                if self._video is not None:
                    self._video.release()
                self._video = cv2.VideoCapture(path) if path else None
                self._video_path = path
            if self._video is None or not self._video.isOpened():
                return default_background(w, h)
            ok, vf = self._video.read()
            if not ok:  # loop
                self._video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, vf = self._video.read()
            return cover(vf, w, h) if ok else default_background(w, h)

        if mode == "screen":
            src = self.screen.latest if self.screen is not None else None
            if src is None:
                return self.placeholder(w, h, "Press 'Share screen / window' to put a chart behind you")
            # re-fit only when the screen actually changed (charts are mostly static)
            key = (getattr(self.screen, "seq", id(src)), w, h, s["screen_fit"])
            if key != self._screen_key:
                self._screen_bg = contain(src, w, h) if s["screen_fit"] else cover(src, w, h)
                self._screen_key = key
            return self._screen_bg

        return frame


# ----------------------------------------------------------------- GUI
import tkinter.font as tkfont
from PIL import ImageDraw, ImageFont

APP_VERSION = "1.0"

# Palette — graphite surfaces, electric-blue accent, broadcast red
C = {
    "bg": "#0a0e13",        # window / preview stage
    "panel": "#10151c",     # sidebar, status bar
    "raised": "#161d27",    # dialogs, menus
    "field": "#18202b",     # controls at rest
    "hover": "#222c39",     # controls on hover
    "line": "#212b37",      # hairlines
    "text": "#e8eef5",
    "muted": "#94a3b3",
    "faint": "#5c6a7a",
    "accent": "#3b82f6",
    "accent_hi": "#6aa5ff",
    "accent_dim": "#14294a",
    "live": "#ef4444",
    "live_hi": "#f87171",
    "live_dim": "#3a1618",
    "ok": "#22c55e",
    "warn": "#f59e0b",
}

FONTS = {}
_IMG = {}
SWATCHES = ["#0b1d2a", "#111827", "#1e293b", "#0f3d3e", "#2e1065",
            "#00b140", "#f1f5f9", "#000000"]


def _rgb(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _cache(key, make):
    if key not in _IMG:
        if len(_IMG) > 800:
            _IMG.clear()
        _IMG[key] = make()
    return _IMG[key]


def setup_fonts(root):
    families = set(tkfont.families(root))
    fam = next((f for f in ("Inter", "Open Sans", "Fira Sans", "Noto Sans", "Ubuntu",
                            "Cantarell", "DejaVu Sans") if f in families), "TkDefaultFont")
    FONTS.update(
        family=fam,
        body=tkfont.Font(root, family=fam, size=-13),
        small=tkfont.Font(root, family=fam, size=-12),
        bold=tkfont.Font(root, family=fam, size=-13, weight="bold"),
        semibold=tkfont.Font(root, family=fam, size=-12, weight="bold"),
        title=tkfont.Font(root, family=fam, size=-16, weight="bold"),
        big=tkfont.Font(root, family=fam, size=-18, weight="bold"),
        hero=tkfont.Font(root, family=fam, size=-20, weight="bold"),
        caps=tkfont.Font(root, family=fam, size=-11, weight="bold"),
        tiny=tkfont.Font(root, family=fam, size=-10, weight="bold"),
    )


def pil_font(size, bold=True):
    """A TrueType font for text drawn into images (badges, toasts)."""
    key = ("pf", size, bold)
    if key in _IMG:
        return _IMG[key]
    path = None
    if shutil.which("fc-match"):
        try:
            q = FONTS.get("family", "DejaVu Sans") + (":bold" if bold else "")
            path = subprocess.run(["fc-match", "-f", "%{file}", q], capture_output=True,
                                  text=True, timeout=3).stdout.strip() or None
        except Exception:
            path = None
    for p in ([path] if path else []) + ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            f = ImageFont.truetype(p, size)
            _IMG[key] = f
            return f
        except Exception:
            continue
    try:
        f = ImageFont.load_default(size=size)
    except TypeError:
        f = ImageFont.load_default()
    _IMG[key] = f
    return f


def ellipsize(font, text, maxw):
    if font.measure(text) <= maxw:
        return text
    while text and font.measure(text + "…") > maxw:
        text = text[:-1]
    return text + "…"


# ---- antialiased drawing (rendered at 4x, downsampled) ----
def rr_image(w, h, r, fill, bg, outline=None, ow=1):
    w, h = max(2, int(w)), max(2, int(h))

    def make():
        S = 4
        img = Image.new("RGB", (w * S, h * S), _rgb(bg))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, w * S - 1, h * S - 1], radius=r * S, fill=_rgb(fill),
                            outline=_rgb(outline) if outline else None,
                            width=ow * S if outline else 0)
        return ImageTk.PhotoImage(img.resize((w, h), Image.LANCZOS))
    return _cache(("rr", w, h, r, fill, bg, outline, ow), make)


def draw_icon(d, name, k, col, lw):
    def P(*pts):
        return [(x * k, y * k) for x, y in zip(pts[0::2], pts[1::2])]

    def line(*pts):
        d.line(P(*pts), fill=col, width=lw, joint="curve")

    def rect(x1, y1, x2, y2, r=2.0, fill=False):
        d.rounded_rectangle([x1 * k, y1 * k, x2 * k, y2 * k], radius=r * k,
                            outline=col, width=lw, fill=col if fill else None)

    def circ(cx, cy, r, fill=False):
        d.ellipse([(cx - r) * k, (cy - r) * k, (cx + r) * k, (cy + r) * k],
                  outline=col, width=0 if fill else lw, fill=col if fill else None)

    def arc(cx, cy, r, a0, a1):
        d.arc([(cx - r) * k, (cy - r) * k, (cx + r) * k, (cy + r) * k], a0, a1, fill=col, width=lw)

    def poly(*pts):
        d.polygon(P(*pts), fill=col)

    if name == "none":
        circ(12, 12, 8.5); line(6, 18, 18, 6)
    elif name == "blur":
        for x, y, r in [(5.5, 5.5, 1.3), (12, 5.5, 1.8), (18.5, 5.5, 1.3), (5.5, 12, 1.8),
                        (12, 12, 2.6), (18.5, 12, 1.8), (5.5, 18.5, 1.3), (12, 18.5, 1.8),
                        (18.5, 18.5, 1.3)]:
            circ(x, y, r, True)
    elif name == "image":
        rect(3, 4.5, 21, 19.5, 2.5); circ(8.8, 9.6, 1.7, True)
        line(3.5, 17.5, 9, 12.5, 13, 16, 16, 13.5, 20.5, 17.5)
    elif name == "video":
        rect(2.5, 5, 21.5, 19, 2.5); poly(10, 8.8, 10, 15.2, 15.6, 12)
    elif name == "colour":
        circ(12, 12, 8.5); circ(12, 12, 5, True)
    elif name == "screen":
        rect(2.5, 4, 21.5, 16, 2); line(8, 20.5, 16, 20.5); line(12, 16, 12, 20.5)
    elif name == "camera":
        rect(2.5, 6.5, 16, 17.5, 2.5); poly(16.5, 10.5, 21.5, 7.5, 21.5, 16.5, 16.5, 13.5)
    elif name == "live":
        circ(12, 12, 2.4, True)
        for r in (6.2, 9.8):
            arc(12, 12, r, -45, 45); arc(12, 12, r, 135, 225)
    elif name == "rec":
        circ(12, 12, 6.5, True)
    elif name == "dot":
        circ(12, 12, 9, True)
    elif name == "stop":
        rect(6.5, 6.5, 17.5, 17.5, 2.2, True)
    elif name == "folder":
        line(3, 7, 3, 18.5, 21, 18.5, 21, 9, 11.5, 9, 9.5, 6, 3.5, 6, 3, 7)
    elif name == "mic":
        rect(9, 3, 15, 14, 3); arc(12, 11, 6.5, 0, 180); line(12, 17.5, 12, 21)
    elif name == "refresh":
        arc(12, 12, 7.5, 40, 330); poly(18.6, 3.8, 20.6, 9.4, 14.6, 8.8)
    elif name == "chevron":
        line(7, 10, 12, 15, 17, 10)
    elif name == "check":
        line(5, 12.5, 10, 17, 19, 7)
    elif name == "alert":
        circ(12, 12, 9.5); line(12, 7, 12, 13.2); circ(12, 16.6, 1.3, True)
    elif name == "info":
        circ(12, 12, 9.5); line(12, 10.8, 12, 17); circ(12, 7.4, 1.3, True)
    elif name == "plus":
        line(12, 5, 12, 19); line(5, 12, 19, 12)
    elif name == "sliders":
        line(4, 7, 20, 7); line(4, 17, 20, 17); circ(9, 7, 2.4, True); circ(15, 17, 2.4, True)


def icon(name, size, color, bg):
    def make():
        S = 4
        px = size * S
        img = Image.new("RGB", (px, px), _rgb(bg))
        d = ImageDraw.Draw(img)
        k = px / 24.0
        draw_icon(d, name, k, _rgb(color), max(1, int(round(2.0 * k))))
        return ImageTk.PhotoImage(img.resize((size, size), Image.LANCZOS))
    return _cache(("ic", name, size, color, bg), make)


def logo_pil(size, bg=None):
    """Nexus 'lens' mark: blue→violet rounded square, white aperture, red on-air dot."""
    S = 4
    px = size * S
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0) if bg is None else _rgb(bg) + (255,))
    grad = Image.new("RGB", (px, px))
    gd = ImageDraw.Draw(grad)
    a, b = _rgb("#2563eb"), _rgb("#7c3aed")
    for y in range(px):
        t = y / max(1, px - 1)
        gd.line([(0, y), (px, y)], fill=tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3)))
    mask = Image.new("L", (px, px), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, px - 1, px - 1], radius=int(px * 0.26), fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    k = px / 24.0
    d.ellipse([6 * k, 6 * k, 18 * k, 18 * k], outline=(255, 255, 255), width=int(2.6 * k))
    d.ellipse([10 * k, 10 * k, 14 * k, 14 * k], fill=(255, 255, 255))
    d.ellipse([17 * k, 3.6 * k, 20.6 * k, 7.2 * k], fill=_rgb("#ff4d4d"))
    return img.resize((size, size), Image.LANCZOS)


def pill_rgba(text, fill, fg, dot=None, alpha=235, size=13, padx=12, h=28):
    """A floating badge (RGBA) for drawing over live video."""
    font = pil_font(size)
    S = 3
    tw = int(font.getlength(text)) if hasattr(font, "getlength") else font.getsize(text)[0]
    dw = 14 if dot else 0
    w = tw + dw + padx * 2
    img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w * S - 1, h * S - 1], radius=h * S // 2, fill=_rgb(fill) + (alpha,))
    img = img.resize((w, h), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    x = padx
    if dot:
        cy = h / 2
        d.ellipse([x, cy - 4, x + 8, cy + 4], fill=_rgb(dot) + (255,))
        x += dw
    asc, desc = font.getmetrics() if hasattr(font, "getmetrics") else (size, 0)
    d.text((x, (h - asc - desc) / 2 + 0.5), text, font=font, fill=_rgb(fg) + (255,))
    return img


# ---- custom widgets ----
class CanvasWidget(tk.Canvas):
    def __init__(self, parent, height, bg, width=10, cursor=""):
        super().__init__(parent, height=height, width=width, bg=bg, highlightthickness=0,
                         bd=0, cursor=cursor)
        self.pbg = bg
        self.hover = False
        self.bind("<Configure>", lambda e: self.redraw())

    def W(self):
        return max(2, self.winfo_width())

    def H(self):
        return max(2, int(self["height"]))

    def redraw(self):
        pass


class Btn(CanvasWidget):
    VARIANTS = {  # rest, hover, text, outline
        "primary": ("accent", "accent_hi", "#ffffff", None),
        "secondary": ("field", "hover", "text", "line"),
        "ghost": (None, "hover", "muted", None),
        "danger": ("live", "live_hi", "#ffffff", None),
        "danger_soft": ("live_dim", "#4a1c1f", "#fca5a5", "#5a2427"),
    }

    def __init__(self, parent, text="", icon_name=None, variant="secondary", command=None,
                 bg=None, height=36, width=None, font=None, radius=8, icon_size=16):
        bg = bg or C["panel"]
        super().__init__(parent, height, bg, width=width or 10, cursor="hand2")
        self.text, self.icon_name, self.variant = text, icon_name, variant
        self.command = command
        self.font = font or FONTS["bold"]
        self.radius, self.icon_size = radius, icon_size
        self.enabled = True
        self._auto_w = width is None
        if self._auto_w:
            self.configure(width=self.natural_width())
        self.bind("<Enter>", lambda e: self._hov(True))
        self.bind("<Leave>", lambda e: self._hov(False))
        self.bind("<ButtonRelease-1>", self._click)

    def natural_width(self):
        tw = self.font.measure(self.text) if self.text else 0
        iw = self.icon_size if self.icon_name else 0
        gap = 8 if tw and iw else 0
        return tw + iw + gap + (30 if tw else 20)

    def _hov(self, v):
        self.hover = v
        self.redraw()

    def _click(self, e):
        if self.enabled and 0 <= e.x <= self.W() and 0 <= e.y <= self.H() and self.command:
            self.command()

    def set(self, text=None, icon_name="__keep__", variant=None, enabled=None):
        before = (self.text, self.icon_name, self.variant, self.enabled)
        if text is not None:
            self.text = text
        if icon_name != "__keep__":
            self.icon_name = icon_name
        if variant is not None:
            self.variant = variant
        if enabled is not None:
            self.enabled = enabled
            self.configure(cursor="hand2" if enabled else "")
        if (self.text, self.icon_name, self.variant, self.enabled) == before:
            return
        if self._auto_w:
            self.configure(width=self.natural_width())
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.W(), self.H()
        rest, hov, fg, outline = self.VARIANTS[self.variant]
        fill = C.get(hov if self.hover else rest) if (hov if self.hover else rest) else self.pbg
        fg = C.get(fg, fg)
        outline = C.get(outline, outline) if outline else None
        if not self.enabled:
            fill, fg, outline = C["field"], C["faint"], None
        self._img = rr_image(w, h, self.radius, fill, self.pbg, outline)
        self.create_image(0, 0, image=self._img, anchor="nw")
        tw = self.font.measure(self.text) if self.text else 0
        iw = self.icon_size if self.icon_name else 0
        gap = 8 if tw and iw else 0
        x = (w - (tw + iw + gap)) / 2
        if iw:
            self._ic = icon(self.icon_name, self.icon_size, fg, fill)
            self.create_image(int(x), h // 2, image=self._ic, anchor="w")
            x += iw + gap
        if tw:
            self.create_text(int(x), h // 2, text=self.text, fill=fg, font=self.font, anchor="w")


class Switch(CanvasWidget):
    def __init__(self, parent, text, variable, command=None, bg=None, sub=None):
        bg = bg or C["panel"]
        super().__init__(parent, 50 if sub else 32, bg, cursor="hand2")
        self.text, self.sub, self.var, self.command = text, sub, variable, command
        variable.trace_add("write", lambda *_: self.redraw())
        self.bind("<Enter>", lambda e: self._hov(True))
        self.bind("<Leave>", lambda e: self._hov(False))
        self.bind("<ButtonRelease-1>", self._click)

    def _hov(self, v):
        self.hover = v
        self.redraw()

    def _click(self, e):
        self.var.set(not bool(self.var.get()))
        if self.command:
            self.command()

    def redraw(self):
        self.delete("all")
        w, h = self.W(), self.H()
        on = bool(self.var.get())
        ty = 16 if not self.sub else 15
        self.create_text(0, ty, text=self.text, fill=C["text"], font=FONTS["body"], anchor="w")
        if self.sub:
            self.create_text(0, 35, text=ellipsize(FONTS["small"], self.sub, w - 60),
                             fill=C["muted"], font=FONTS["small"], anchor="w")
        key = ("sw", on, self.pbg, self.hover)

        def make():
            sw, sh, S = 38, 22, 4
            img = Image.new("RGB", (sw * S, sh * S), _rgb(self.pbg))
            d = ImageDraw.Draw(img)
            track = C["accent"] if on else (C["faint"] if self.hover else C["hover"])
            d.rounded_rectangle([0, 0, sw * S - 1, sh * S - 1], radius=sh * S // 2, fill=_rgb(track))
            kx = (sw - sh) * S if on else 0
            pad = 3 * S
            d.ellipse([kx + pad, pad, kx + sh * S - pad, sh * S - pad], fill=(255, 255, 255))
            return ImageTk.PhotoImage(img.resize((sw, sh), Image.LANCZOS))
        self._img = _cache(key, make)
        self.create_image(w, ty, image=self._img, anchor="e")


class Slider(CanvasWidget):
    def __init__(self, parent, label, variable, lo, hi, command=None, bg=None, suffix=""):
        bg = bg or C["panel"]
        super().__init__(parent, 50, bg, cursor="hand2")
        self.label, self.var, self.lo, self.hi = label, variable, lo, hi
        self.command, self.suffix = command, suffix
        self.active = False
        variable.trace_add("write", lambda *_: self.redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Button-4>", lambda e: self._step(1))
        self.bind("<Button-5>", lambda e: self._step(-1))
        self.bind("<MouseWheel>", lambda e: self._step(1 if e.delta > 0 else -1))
        self.bind("<Enter>", lambda e: self._hov(True))
        self.bind("<Leave>", lambda e: self._hov(False))

    def _hov(self, v):
        self.hover = v
        self.redraw()

    def _set_from_x(self, x):
        pad = 9
        frac = min(1.0, max(0.0, (x - pad) / max(1, self.W() - 2 * pad)))
        v = int(round(self.lo + frac * (self.hi - self.lo)))
        if v != int(self.var.get()):
            self.var.set(v)
            if self.command:
                self.command()

    def _press(self, e):
        self.active = True
        self._set_from_x(e.x)
        self.redraw()

    def _drag(self, e):
        self._set_from_x(e.x)

    def _release(self, e):
        self.active = False
        self.redraw()

    def _step(self, d):
        v = min(self.hi, max(self.lo, int(self.var.get()) + d))
        self.var.set(v)
        if self.command:
            self.command()

    def redraw(self):
        self.delete("all")
        w = self.W()
        v = int(self.var.get())
        self.create_text(0, 12, text=self.label, fill=C["text"], font=FONTS["body"], anchor="w")
        self.create_text(w, 12, text=f"{v}{self.suffix}", fill=C["accent_hi"] if self.active else C["muted"],
                         font=FONTS["semibold"], anchor="e")
        frac = (v - self.lo) / max(1, self.hi - self.lo)
        S, th = 4, 22
        img = Image.new("RGB", (w * S, th * S), _rgb(self.pbg))
        d = ImageDraw.Draw(img)
        pad = 9 * S
        cy = th * S // 2
        x0, x1 = pad, w * S - pad
        kx = x0 + frac * (x1 - x0)
        d.rounded_rectangle([x0, cy - 2 * S, x1, cy + 2 * S], radius=2 * S, fill=_rgb(C["hover"]))
        if kx > x0:
            d.rounded_rectangle([x0, cy - 2 * S, kx, cy + 2 * S], radius=2 * S, fill=_rgb(C["accent"]))
        kr = (8 if (self.active or self.hover) else 7) * S
        if self.active:
            d.ellipse([kx - kr - 3 * S, cy - kr - 3 * S, kx + kr + 3 * S, cy + kr + 3 * S],
                      fill=_rgb(C["accent_dim"]))
        d.ellipse([kx - kr, cy - kr, kx + kr, cy + kr], fill=(255, 255, 255))
        self._img = ImageTk.PhotoImage(img.resize((w, th), Image.LANCZOS))
        self.create_image(0, 28, image=self._img, anchor="nw")


class Segmented(CanvasWidget):
    def __init__(self, parent, options, variable, command=None, bg=None, height=34):
        bg = bg or C["panel"]
        super().__init__(parent, height, bg, cursor="hand2")
        self.options, self.var, self.command = options, variable, command
        self._hover_i = -1
        variable.trace_add("write", lambda *_: self.redraw())
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", lambda e: self._set_hover(-1))
        self.bind("<ButtonRelease-1>", self._click)

    def _index_at(self, x):
        n = len(self.options)
        return min(n - 1, max(0, int((x - 3) / max(1, (self.W() - 6) / n))))

    def _set_hover(self, i):
        if i != self._hover_i:
            self._hover_i = i
            self.redraw()

    def _motion(self, e):
        self._set_hover(self._index_at(e.x))

    def _click(self, e):
        val = self.options[self._index_at(e.x)][0]
        if val != self.var.get():
            self.var.set(val)
            if self.command:
                self.command()

    def redraw(self):
        self.delete("all")
        w, h = self.W(), self.H()
        self._bg = rr_image(w, h, 9, C["field"], self.pbg, C["line"])
        self.create_image(0, 0, image=self._bg, anchor="nw")
        n = len(self.options)
        segw = (w - 6) / n
        self._segs = []
        for i, (val, label) in enumerate(self.options):
            x0 = 3 + i * segw
            sel = self.var.get() == val
            if sel or i == self._hover_i:
                fill = C["accent"] if sel else C["hover"]
                img = rr_image(segw, h - 6, 7, fill, C["field"])
                self._segs.append(img)
                self.create_image(int(x0), 3, image=img, anchor="nw")
            self.create_text(int(x0 + segw / 2), h // 2, text=label,
                             fill="#ffffff" if sel else (C["text"] if i == self._hover_i else C["muted"]),
                             font=FONTS["semibold"])


class Dropdown(CanvasWidget):
    def __init__(self, parent, command=None, bg=None, height=38, placeholder="—"):
        bg = bg or C["panel"]
        super().__init__(parent, height, bg, cursor="hand2")
        self.values, self.index = [], -1
        self.command, self.placeholder = command, placeholder
        self.bind("<Enter>", lambda e: self._hov(True))
        self.bind("<Leave>", lambda e: self._hov(False))
        self.bind("<ButtonRelease-1>", self._open)

    def _hov(self, v):
        self.hover = v
        self.redraw()

    def set_values(self, values, index=0):
        self.values = list(values)
        self.index = index if 0 <= index < len(self.values) else (0 if self.values else -1)
        self.redraw()

    def current(self):
        return self.index

    def _open(self, e):
        if not self.values:
            return
        m = tk.Menu(self, tearoff=0, bg=C["raised"], fg=C["text"], activebackground=C["accent"],
                    activeforeground="#ffffff", bd=0, relief="flat", font=FONTS["body"],
                    activeborderwidth=6)
        for i, v in enumerate(self.values):
            m.add_command(label=("✓  " if i == self.index else "     ") + v,
                          command=lambda i=i: self._choose(i))
        m.tk_popup(self.winfo_rootx(), self.winfo_rooty() + self.H() + 4)

    def _choose(self, i):
        changed = i != self.index
        self.index = i
        self.redraw()
        if changed and self.command:
            self.command()

    def redraw(self):
        self.delete("all")
        w, h = self.W(), self.H()
        self._bg = rr_image(w, h, 9, C["field"], self.pbg, C["faint"] if self.hover else C["line"])
        self.create_image(0, 0, image=self._bg, anchor="nw")
        txt = self.values[self.index] if 0 <= self.index < len(self.values) else self.placeholder
        self.create_text(12, h // 2, text=ellipsize(FONTS["body"], txt, w - 44),
                         fill=C["text"] if self.values else C["faint"], font=FONTS["body"], anchor="w")
        self._ic = icon("chevron", 16, C["muted"], C["field"])
        self.create_image(w - 12, h // 2, image=self._ic, anchor="e")


class ModeTiles(tk.Frame):
    ICONS = {"none": "none", "blur": "blur", "image": "image", "video": "video",
             "colour": "colour", "screen": "screen"}

    def __init__(self, parent, variable, command, bg=None):
        bg = bg or C["panel"]
        super().__init__(parent, bg=bg)
        self.tiles = []
        for i, m in enumerate(MODES):
            t = _Tile(self, m, MODE_LABELS[m], self.ICONS[m], i + 1, variable, command, bg)
            t.grid(row=i // 3, column=i % 3, sticky="ew",
                   padx=(0 if i % 3 == 0 else 4, 0 if i % 3 == 2 else 4), pady=4)
            self.tiles.append(t)
        for c in range(3):
            self.grid_columnconfigure(c, weight=1, uniform="tiles")


class _Tile(CanvasWidget):
    def __init__(self, parent, value, label, icon_name, key, variable, command, bg):
        super().__init__(parent, 72, bg, cursor="hand2")
        self.value, self.label, self.icon_name, self.key = value, label, icon_name, key
        self.var, self.command = variable, command
        variable.trace_add("write", lambda *_: self.redraw())
        self.bind("<Enter>", lambda e: self._hov(True))
        self.bind("<Leave>", lambda e: self._hov(False))
        self.bind("<ButtonRelease-1>", lambda e: self.command(self.value))

    def _hov(self, v):
        self.hover = v
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.W(), self.H()
        sel = self.var.get() == self.value
        fill = C["accent_dim"] if sel else (C["hover"] if self.hover else C["field"])
        self._bg = rr_image(w, h, 10, fill, self.pbg, C["accent"] if sel else C["line"])
        self.create_image(0, 0, image=self._bg, anchor="nw")
        col = C["accent_hi"] if sel else (C["text"] if self.hover else C["muted"])
        self._ic = icon(self.icon_name, 22, col, fill)
        self.create_image(w // 2, 29, image=self._ic)
        self.create_text(w // 2, 55, text=self.label, font=FONTS["semibold"],
                         fill=C["text"] if sel else (C["text"] if self.hover else C["muted"]))
        self.create_text(w - 9, 10, text=str(self.key), font=FONTS["tiny"], fill=C["faint"], anchor="ne")


class Swatches(CanvasWidget):
    def __init__(self, parent, variable, command, bg=None):
        bg = bg or C["panel"]
        super().__init__(parent, 34, bg, cursor="hand2")
        self.var, self.command = variable, command
        variable.trace_add("write", lambda *_: self.redraw())
        self.bind("<ButtonRelease-1>", self._click)

    def _geom(self):
        n = len(SWATCHES)
        step = min(36, (self.W()) / n)
        return n, step

    def _click(self, e):
        n, step = self._geom()
        i = int(e.x // step)
        if 0 <= i < n:
            self.command(SWATCHES[i])

    def redraw(self):
        self.delete("all")
        w, h = self.W(), self.H()
        n, step = self._geom()
        S = 4
        img = Image.new("RGB", (w * S, h * S), _rgb(self.pbg))
        d = ImageDraw.Draw(img)
        cur = self.var.get().lower()
        for i, c in enumerate(SWATCHES):
            cx, cy, r = (i * step + step / 2) * S, h / 2 * S, 12 * S
            if c.lower() == cur:
                d.ellipse([cx - r - 3 * S, cy - r - 3 * S, cx + r + 3 * S, cy + r + 3 * S],
                          outline=_rgb(C["accent"]), width=2 * S)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_rgb(c), outline=_rgb(C["line"]), width=S)
        self._img = ImageTk.PhotoImage(img.resize((w, h), Image.LANCZOS))
        self.create_image(0, 0, image=self._img, anchor="nw")


class Chip(CanvasWidget):
    STYLES = {  # fill, text, dot, outline
        "off": ("field", "muted", "faint", "line"),
        "on": ("field", "text", "ok", "line"),
        "live": ("live", "#ffffff", "#ffffff", None),
        "rec": ("live_dim", "#fecaca", "live", "#5a2427"),
    }

    def __init__(self, parent, bg):
        super().__init__(parent, 28, bg)
        self.text, self.kind = "", "off"
        self.visible = True

    def show(self, v):
        if v != self.visible:
            self.visible = v
            if v:
                self.pack(side="right", padx=(8, 0))
            else:
                self.pack_forget()

    def set(self, text, kind):
        if (text, kind) == (self.text, self.kind):
            return
        self.text, self.kind = text, kind
        self.configure(width=FONTS["semibold"].measure(text) + 36)
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.W(), self.H()
        fill, fg, dot, outline = (C.get(x, x) if x else None for x in self.STYLES[self.kind])
        self._bg = rr_image(w, h, 14, fill, self.pbg, outline)
        self.create_image(0, 0, image=self._bg, anchor="nw")
        self._dot = icon("dot", 8, dot, fill)
        self.create_image(13, h // 2, image=self._dot)
        self.create_text(23, h // 2, text=self.text, font=FONTS["semibold"], fill=fg, anchor="w")


class LevelMeter(CanvasWidget):
    def __init__(self, parent, bg=None):
        super().__init__(parent, 10, bg or C["panel"])
        self.db = None

    def set_level(self, db):
        self.db = db
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.W(), self.H()
        n, gap = 32, 3
        sw = (w - gap * (n - 1)) / n
        lit = 0 if self.db is None else int(round(min(1.0, max(0.0, (self.db + 60) / 60)) * n))
        for i in range(n):
            x = i * (sw + gap)
            frac = (i + 1) / n
            if i < lit:
                col = C["ok"] if frac < 0.8 else (C["warn"] if frac < 0.95 else C["live"])
            else:
                col = C["hover"]
            self.create_rectangle(x, 0, x + sw, h, fill=col, width=0)


# ---- app ----
class App:
    def __init__(self, root):
        self.root = root
        root.title("NexusCam")
        root.configure(bg=C["bg"])
        root.geometry("1400x860")
        root.minsize(1120, 720)
        os.makedirs(APP_DIR, exist_ok=True)
        setup_fonts(root)
        root.option_add("*Font", FONTS["body"])
        self.s = self.load_settings()
        self.engine = None
        self.screen = ScreenSource(self.s.get("screen_token", ""))
        self.recorder = None
        self._rec_pending = False
        self._last_saved = None
        self.cams, self.mics = [], []
        self.meter = None
        self._shown_frame = None
        self._photo = None
        self._masks = {}
        self._badges = {}
        self._toast = None
        self._toast_after = None
        self._was_live = False
        self._last_screen_status = None

        self.make_vars()
        self.build()
        self.refresh_cameras()
        self.refresh_mics()
        self.bind_keys()
        self.show_context()
        self.push()
        self.set_icons()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.tick()

    # ---------------------------------------------------------- settings
    def load_settings(self):
        s = dict(DEFAULTS)
        try:
            with open(SETTINGS_FILE) as f:
                s.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
        except (OSError, ValueError):
            pass
        return s

    def save_settings(self):
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(self.collect(), f, indent=2)
        except OSError:
            pass

    def make_vars(self):
        s = self.s
        self.v_res = tk.StringVar(value=s["resolution"] if s["resolution"] in RESOLUTIONS else "1280x720")
        self.v_out = tk.StringVar(value=s["output_size"] if s["output_size"] in OUTPUT_SIZES else OUTPUT_SIZES[0])
        self.v_mode = tk.StringVar(value=s["mode"] if s["mode"] in MODES else "blur")
        self.v_blur = tk.IntVar(value=s["blur"])
        self.v_soft = tk.IntVar(value=s["softness"])
        self.v_thr = tk.IntVar(value=s["threshold"])
        self.v_smooth = tk.IntVar(value=s["smoothing"])
        self.v_mirror = tk.BooleanVar(value=s["mirror"])
        self.v_invert = tk.BooleanVar(value=s["invert"])
        self.v_image = tk.StringVar(value=s["image_path"])
        self.v_video = tk.StringVar(value=s["video_path"])
        self.v_colour = tk.StringVar(value=s["colour"])
        self.v_vdev = tk.StringVar(value=s["vcam_device"])
        self.v_fit = tk.BooleanVar(value=s["screen_fit"])
        self.v_pos = tk.StringVar(value=s["presenter_pos"] if s["presenter_pos"] in POSITIONS else "Centre")
        self.v_psize = tk.IntVar(value=s["presenter_size"])
        self.v_mic = tk.BooleanVar(value=s["record_audio"])
        self.v_recdir = tk.StringVar(value=s.get("record_dir") or DEFAULTS["record_dir"])
        self.v_tab = tk.StringVar(value="scene")
        for v in (self.v_blur, self.v_soft, self.v_thr, self.v_smooth, self.v_mirror, self.v_invert,
                  self.v_image, self.v_video, self.v_colour, self.v_vdev, self.v_fit, self.v_pos,
                  self.v_psize):
            v.trace_add("write", lambda *_: self.push())

    def collect(self):
        return {
            "camera": self.cam_path(),
            "resolution": self.v_res.get(),
            "fps": DEFAULTS["fps"],
            "mode": self.v_mode.get(),
            "blur": int(self.v_blur.get()),
            "softness": int(self.v_soft.get()),
            "threshold": int(self.v_thr.get()),
            "smoothing": int(self.v_smooth.get()),
            "mirror": bool(self.v_mirror.get()),
            "invert": bool(self.v_invert.get()),
            "image_path": self.v_image.get(),
            "video_path": self.v_video.get(),
            "colour": self.v_colour.get(),
            "vcam_device": self.v_vdev.get().strip() or DEFAULTS["vcam_device"],
            "screen_fit": bool(self.v_fit.get()),
            "screen_token": self.screen.restore_token or "",
            "presenter_pos": self.v_pos.get(),
            "presenter_size": int(self.v_psize.get()),
            "output_size": self.v_out.get(),
            "record_audio": bool(self.v_mic.get()),
            "mic_source": self.mic_name(),
            "record_dir": self.v_recdir.get() or DEFAULTS["record_dir"],
        }

    def push(self, *_):
        if self.engine is not None:
            self.engine.update(self.collect())

    # ---------------------------------------------------------- layout helpers
    def caps(self, parent, text, first=False):
        tk.Label(parent, text=text.upper(), font=FONTS["caps"], fg=C["faint"], bg=C["panel"]
                 ).pack(anchor="w", pady=(6 if first else 22, 10))

    def hint(self, parent, text, pady=(8, 0)):
        lbl = tk.Label(parent, text=text, font=FONTS["small"], fg=C["muted"], bg=C["panel"],
                       wraplength=340, justify="left")
        lbl.pack(anchor="w", pady=pady)
        return lbl

    def row(self, parent, pady=(0, 0)):
        f = tk.Frame(parent, bg=C["panel"])
        f.pack(fill="x", pady=pady)
        return f

    # ---------------------------------------------------------- build
    def build(self):
        root = self.root

        # header
        hdr = tk.Frame(root, bg=C["bg"], height=62)
        hdr.pack(side="top", fill="x")
        hdr.pack_propagate(False)
        self._logo = ImageTk.PhotoImage(logo_pil(32, C["bg"]).convert("RGB"))
        tk.Label(hdr, image=self._logo, bg=C["bg"]).pack(side="left", padx=(22, 12))
        tb = tk.Frame(hdr, bg=C["bg"])
        tb.pack(side="left")
        tk.Label(tb, text="NexusCam", font=FONTS["title"], fg=C["text"], bg=C["bg"]).pack(anchor="w")
        tk.Label(tb, text="NEXUS TRADING SYSTEMS", font=FONTS["tiny"], fg=C["faint"],
                 bg=C["bg"]).pack(anchor="w")
        chips = tk.Frame(hdr, bg=C["bg"])
        chips.pack(side="right", padx=22)
        self.chip_cam = Chip(chips, C["bg"])
        self.chip_live = Chip(chips, C["bg"])
        self.chip_rec = Chip(chips, C["bg"])
        for c in (self.chip_cam, self.chip_live, self.chip_rec):
            c.pack(side="left", padx=(8, 0))
        self.chip_live.show = lambda v, c=self.chip_live: self._chip_show(c, v)
        self.chip_rec.show = lambda v, c=self.chip_rec: self._chip_show(c, v)
        self._chip_show(self.chip_live, False)
        self._chip_show(self.chip_rec, False)
        tk.Frame(root, bg=C["line"], height=1).pack(side="top", fill="x")

        # status bar
        sb = tk.Frame(root, bg=C["panel"], height=32)
        sb.pack(side="bottom", fill="x")
        sb.pack_propagate(False)
        tk.Frame(root, bg=C["line"], height=1).pack(side="bottom", fill="x")
        self.lbl_stats = tk.Label(sb, text="Camera off", font=FONTS["small"], fg=C["muted"], bg=C["panel"])
        self.lbl_stats.pack(side="left", padx=18)
        keys = tk.Frame(sb, bg=C["panel"])
        keys.pack(side="right", padx=18)
        for k, label in (("1–6", "Background"), ("Space", "Live"), ("R", "Record")):
            tk.Label(keys, text=f" {k} ", font=FONTS["tiny"], fg=C["text"], bg=C["hover"]
                     ).pack(side="left", padx=(12, 5))
            tk.Label(keys, text=label, font=FONTS["small"], fg=C["muted"], bg=C["panel"]).pack(side="left")
        self.lbl_enc = tk.Label(sb, text="", font=FONTS["small"], fg=C["faint"], bg=C["panel"])
        self.lbl_enc.pack(side="right", padx=6)

        body = tk.Frame(root, bg=C["bg"])
        body.pack(fill="both", expand=True)

        # sidebar
        side = tk.Frame(body, bg=C["panel"], width=392)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        tk.Frame(body, bg=C["line"], width=1).pack(side="right", fill="y")
        self.tabs = Segmented(side, [("scene", "Scene"), ("cutout", "Cut-out"),
                                     ("audio", "Audio"), ("setup", "Setup")],
                              self.v_tab, command=self.show_tab, height=38)
        self.tabs.pack(fill="x", padx=22, pady=(20, 8))
        self.pages = {}
        for k in ("scene", "cutout", "audio", "setup"):
            self.pages[k] = tk.Frame(side, bg=C["panel"])
        self.build_scene(self.pages["scene"])
        self.build_cutout(self.pages["cutout"])
        self.build_audio(self.pages["audio"])
        self.build_setup(self.pages["setup"])
        self.show_tab()

        # main stage
        main = tk.Frame(body, bg=C["bg"])
        main.pack(side="left", fill="both", expand=True)
        transport = tk.Frame(main, bg=C["bg"], height=84)
        transport.pack(side="bottom", fill="x")
        transport.pack_propagate(False)
        self.preview = tk.Canvas(main, bg=C["bg"], highlightthickness=0, bd=0)
        self.preview.pack(fill="both", expand=True, padx=24, pady=(22, 0))
        self.preview.bind("<Configure>", lambda e: self.render_preview(force=True))
        inner = tk.Frame(transport, bg=C["bg"])
        inner.place(relx=0.5, rely=0.5, anchor="center")
        self.btn_cam = Btn(inner, "Start camera", "camera", "secondary", self.toggle_camera,
                           bg=C["bg"], width=172, height=44, radius=10, icon_size=18)
        self.btn_live = Btn(inner, "Go live", "live", "primary", self.toggle_live,
                            bg=C["bg"], width=172, height=44, radius=10, icon_size=18)
        self.btn_rec = Btn(inner, "Record", "rec", "danger_soft", self.toggle_record,
                           bg=C["bg"], width=172, height=44, radius=10, icon_size=18)
        for b in (self.btn_cam, self.btn_live, self.btn_rec):
            b.pack(side="left", padx=7)
        self.start_overlay = Btn(self.preview, "Start camera", "camera", "primary", self.toggle_camera,
                                 bg=C["bg"], width=180, height=44, radius=10, icon_size=18)

    def build_scene(self, p):
        p = tk.Frame(p, bg=C["panel"])
        p.pack(fill="both", expand=True, padx=22)
        self.caps(p, "Background", first=True)
        ModeTiles(p, self.v_mode, self.set_mode).pack(fill="x")
        self.ctx_holder = tk.Frame(p, bg=C["panel"])
        self.ctx_holder.pack(fill="x", pady=(14, 0))
        self.ctx = {m: tk.Frame(self.ctx_holder, bg=C["panel"]) for m in MODES}

        self.hint(self.ctx["none"], "Your camera exactly as it is — no cut-out, lowest CPU use.", pady=(0, 0))

        Slider(self.ctx["blur"], "Blur strength", self.v_blur, 5, 99).pack(fill="x")

        f = self.ctx["image"]
        r = self.row(f)
        self.img_thumb = tk.Label(r, bg=C["panel"])
        self.img_thumb.pack(side="left")
        col = tk.Frame(r, bg=C["panel"])
        col.pack(side="left", fill="x", expand=True, padx=(12, 0))
        self.lbl_img = tk.Label(col, text="", font=FONTS["bold"], fg=C["text"], bg=C["panel"], anchor="w")
        self.lbl_img.pack(fill="x")
        self.lbl_img_sub = tk.Label(col, text="", font=FONTS["small"], fg=C["muted"], bg=C["panel"], anchor="w")
        self.lbl_img_sub.pack(fill="x")
        r2 = self.row(f, pady=(12, 0))
        Btn(r2, "Choose image…", "image", "secondary", self.pick_image).pack(side="left")
        self.btn_img_default = Btn(r2, "Use Nexus backdrop", None, "ghost", self.use_default_image)
        self.btn_img_default.pack(side="left", padx=(8, 0))

        f = self.ctx["video"]
        r = self.row(f)
        self._vid_icon = icon("video", 22, C["muted"], C["field"])
        tk.Label(r, image=self._vid_icon, bg=C["field"], width=64, height=38).pack(side="left")
        col = tk.Frame(r, bg=C["panel"])
        col.pack(side="left", fill="x", expand=True, padx=(12, 0))
        self.lbl_vid = tk.Label(col, text="", font=FONTS["bold"], fg=C["text"], bg=C["panel"], anchor="w")
        self.lbl_vid.pack(fill="x")
        tk.Label(col, text="Loops behind you", font=FONTS["small"], fg=C["muted"], bg=C["panel"],
                 anchor="w").pack(fill="x")
        Btn(self.row(f, pady=(12, 0)), "Choose video…", "video", "secondary", self.pick_video).pack(side="left")

        f = self.ctx["colour"]
        Swatches(f, self.v_colour, self.set_colour).pack(fill="x")
        r = self.row(f, pady=(10, 0))
        Btn(r, "Custom colour…", "plus", "ghost", self.pick_colour).pack(side="left")
        self.lbl_colour = tk.Label(r, text="", font=FONTS["small"], fg=C["muted"], bg=C["panel"])
        self.lbl_colour.pack(side="right")

        f = self.ctx["screen"]
        r = self.row(f)
        self.scr_dot = tk.Label(r, text="●", font=FONTS["body"], fg=C["faint"], bg=C["panel"])
        self.scr_dot.pack(side="left")
        self.scr_status = tk.Label(r, text="Not sharing", font=FONTS["bold"], fg=C["text"], bg=C["panel"])
        self.scr_status.pack(side="left", padx=(6, 0))
        r2 = self.row(f, pady=(12, 0))
        self.btn_share = Btn(r2, "Share a window", "screen", "primary", self.toggle_screen, width=190)
        self.btn_share.pack(side="left")
        Btn(r2, "Change", None, "ghost", self.change_screen).pack(side="left", padx=(8, 0))
        Switch(f, "Fit whole window", self.v_fit, sub="Shows the full chart, bars at the edges").pack(
            fill="x", pady=(12, 0))

        self.caps(p, "Presenter")
        Segmented(p, [(x, x) for x in POSITIONS], self.v_pos).pack(fill="x")
        Slider(p, "Size", self.v_psize, 40, 100, suffix="%").pack(fill="x", pady=(10, 0))
        self.hint(p, "Shrink yourself and move aside so the chart stays visible. "
                     "Not applied with Blur.")

    def build_cutout(self, p):
        p = tk.Frame(p, bg=C["panel"])
        p.pack(fill="both", expand=True, padx=22)
        self.caps(p, "Edge quality", first=True)
        Slider(p, "Edge softness", self.v_soft, 0, 20).pack(fill="x")
        Slider(p, "Cut-out tightness", self.v_thr, 20, 80).pack(fill="x", pady=(8, 0))
        Slider(p, "Flicker smoothing", self.v_smooth, 0, 100).pack(fill="x", pady=(8, 0))
        Switch(p, "Invert mask", self.v_invert, sub="Use if the background shows instead of you").pack(
            fill="x", pady=(12, 0))
        Btn(self.row(p, pady=(14, 0)), "Reset to defaults", "refresh", "ghost", self.reset_cutout).pack(side="left")
        self.caps(p, "Tips")
        tip = tk.Frame(p, bg=C["field"], highlightthickness=1, highlightbackground=C["line"])
        tip.pack(fill="x")
        self._tip_ic = icon("info", 18, C["accent_hi"], C["field"])
        tk.Label(tip, image=self._tip_ic, bg=C["field"]).pack(side="left", anchor="n", padx=(12, 0), pady=12)
        tk.Label(tip, text="Light on your face does more than any slider. A plain wall behind you "
                           "and 60–100 cm from the camera give the cleanest edges. Raise tightness "
                           "if a halo appears around your head.",
                 font=FONTS["small"], fg=C["muted"], bg=C["field"], wraplength=290, justify="left"
                 ).pack(side="left", padx=10, pady=12)

    def build_audio(self, p):
        p = tk.Frame(p, bg=C["panel"])
        p.pack(fill="both", expand=True, padx=22)
        self.caps(p, "Microphone", first=True)
        Switch(p, "Record microphone", self.v_mic, sub="Your voice in the recorded video").pack(fill="x")
        tk.Label(p, text="Input", font=FONTS["body"], fg=C["text"], bg=C["panel"]).pack(anchor="w", pady=(14, 6))
        r = self.row(p)
        self.dd_mic = Dropdown(r, command=self.on_mic_change, placeholder="System default input")
        self.dd_mic.pack(side="left", fill="x", expand=True)
        Btn(r, "", "refresh", "secondary", self.refresh_mics, width=38, height=38).pack(side="left", padx=(8, 0))
        self.caps(p, "Level")
        self.level = LevelMeter(p)
        self.level.pack(fill="x")
        r = self.row(p, pady=(12, 0))
        self.lbl_meter = tk.Label(r, text="Press Test, then speak.",
                                  font=FONTS["small"], fg=C["muted"], bg=C["panel"])
        self.lbl_meter.pack(side="left")
        self.btn_test = Btn(r, "Test", "mic", "secondary", self.toggle_meter, width=88)
        self.btn_test.pack(side="right")
        self.hint(p, "“Computer audio” inputs record what your laptop plays — TradingView "
                     "alerts, for example — instead of your voice.", pady=(18, 0))

    def build_setup(self, p):
        p = tk.Frame(p, bg=C["panel"])
        p.pack(fill="both", expand=True, padx=22)
        self.caps(p, "Camera", first=True)
        r = self.row(p)
        self.dd_cam = Dropdown(r, command=self.restart_if_running, placeholder="No webcam found")
        self.dd_cam.pack(side="left", fill="x", expand=True)
        Btn(r, "", "refresh", "secondary", self.refresh_cameras, width=38, height=38).pack(side="left", padx=(8, 0))
        tk.Label(p, text="Camera resolution", font=FONTS["body"], fg=C["text"], bg=C["panel"]
                 ).pack(anchor="w", pady=(14, 6))
        Segmented(p, [("640x480", "480p"), ("1280x720", "720p"), ("1920x1080", "1080p")],
                  self.v_res, command=self.restart_if_running).pack(fill="x")
        tk.Label(p, text="Output size", font=FONTS["body"], fg=C["text"], bg=C["panel"]
                 ).pack(anchor="w", pady=(14, 6))
        Segmented(p, [("Same as camera", "Camera"), ("1280x720", "720p"), ("1920x1080", "1080p")],
                  self.v_out, command=self.restart_if_running).pack(fill="x")
        self.hint(p, "1080p output keeps charts sharp even with a 720p webcam.")
        Switch(p, "Mirror me", self.v_mirror, sub="Only you are flipped — charts stay readable").pack(
            fill="x", pady=(10, 0))

        self.caps(p, "Virtual camera")
        box = tk.Frame(p, bg=C["field"], highlightthickness=1, highlightbackground=C["line"],
                       highlightcolor=C["accent"])
        box.pack(fill="x")
        tk.Entry(box, textvariable=self.v_vdev, font=FONTS["body"], bg=C["field"], fg=C["text"],
                 insertbackground=C["text"], relief="flat", bd=0, highlightthickness=0
                 ).pack(fill="x", padx=12, pady=9)
        self.hint(p, "In Zoom, Meet, OBS or your browser, choose “NexusCam” as the camera.")

        self.caps(p, "Recordings")
        r = self.row(p)
        self.lbl_recdir = tk.Label(r, text="", font=FONTS["small"], fg=C["muted"], bg=C["panel"], anchor="w")
        self.lbl_recdir.pack(side="left", fill="x", expand=True)
        r2 = self.row(p, pady=(10, 0))
        Btn(r2, "Change…", None, "secondary", self.change_rec_dir).pack(side="left")
        Btn(r2, "Open folder", "folder", "ghost", self.open_rec_folder).pack(side="left", padx=(8, 0))
        tk.Label(p, text=f"NexusCam {APP_VERSION}  ·  Nexus Trading Systems", font=FONTS["small"],
                 fg=C["faint"], bg=C["panel"]).pack(anchor="w", pady=(26, 0))
        self.update_recdir_label()

    def _chip_show(self, chip, v):
        if v != chip.visible:
            chip.visible = v
            if v:
                chip.pack(side="left", padx=(8, 0))
            else:
                chip.pack_forget()

    # ---------------------------------------------------------- navigation
    def show_tab(self):
        for k, f in self.pages.items():
            f.pack_forget()
        self.pages[self.v_tab.get()].pack(fill="both", expand=True)
        if self.v_tab.get() != "audio" and self.meter is not None:
            self.toggle_meter()

    def show_context(self):
        for f in self.ctx.values():
            f.pack_forget()
        m = self.v_mode.get()
        self.ctx[m].pack(fill="x")
        self.update_context()

    def update_context(self):
        m = self.v_mode.get()
        if m == "image":
            p = self.v_image.get()
            self.lbl_img.configure(text=ellipsize(FONTS["bold"], os.path.basename(p), 230) if p else "Nexus backdrop")
            self.lbl_img_sub.configure(text="Your image" if p else "Built-in default")
            self._thumb = self.make_thumb(p)
            self.img_thumb.configure(image=self._thumb)
            self.btn_img_default.set(enabled=bool(p))
        elif m == "video":
            p = self.v_video.get()
            self.lbl_vid.configure(text=ellipsize(FONTS["bold"], os.path.basename(p), 230) if p else "No video chosen")
        elif m == "colour":
            self.lbl_colour.configure(text=self.v_colour.get().upper())

    def make_thumb(self, path):
        tw, th = 96, 54
        img = cv2.imread(path) if path else None
        if img is None:
            img = default_background(1280, 720)
        img = cv2.cvtColor(cover(img, tw * 2, th * 2), cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img).resize((tw * 2, th * 2))
        base = Image.new("RGB", (tw * 2, th * 2), _rgb(C["panel"]))
        mask = Image.new("L", (tw * 2, th * 2), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw * 2 - 1, th * 2 - 1], radius=14, fill=255)
        base.paste(pil, (0, 0), mask)
        return ImageTk.PhotoImage(base.resize((tw, th), Image.LANCZOS))

    # ---------------------------------------------------------- devices
    def refresh_cameras(self):
        vdev = self.v_vdev.get().strip()
        self.cams = [(p, n) for p, n in list_cameras() if not is_virtual(n, p, vdev)]
        saved = self.s.get("camera")
        idx = next((i for i, (p, _) in enumerate(self.cams) if p == saved), 0)
        self.dd_cam.set_values([n for _, n in self.cams], idx)

    def cam_path(self):
        i = self.dd_cam.current() if hasattr(self, "dd_cam") else -1
        return self.cams[i][0] if 0 <= i < len(self.cams) else ""

    def refresh_mics(self):
        self.mics = list_mic_sources()
        saved = self.s.get("mic_source", "")
        idx = next((i + 1 for i, (n, _) in enumerate(self.mics) if n == saved), 0)
        self.dd_mic.set_values(["System default input"] + [lbl for _, lbl in self.mics], idx)
        if not self.mics and not shutil.which("pactl"):
            self.lbl_meter.configure(text="Run setup.sh to list inputs.")

    def mic_name(self):
        i = self.dd_mic.current() if hasattr(self, "dd_mic") else 0
        return self.mics[i - 1][0] if 1 <= i <= len(self.mics) else ""

    def on_mic_change(self):
        self.s["mic_source"] = self.mic_name()
        if self.meter is not None:
            self.toggle_meter()
            self.toggle_meter()

    def toggle_meter(self):
        if self.meter is not None:
            self.meter.stop()
            self.meter = None
            self.btn_test.set(text="Test", icon_name="mic")
            self.level.set_level(None)
            self.lbl_meter.configure(text="Press Test, then speak.", fg=C["muted"])
        else:
            if shutil.which("ffmpeg") is None:
                self.alert("ffmpeg is missing", "Run setup.sh to install it.")
                return
            self.meter = MicMeter(self.mic_name())
            self.btn_test.set(text="Stop", icon_name="stop")

    def draw_meter(self):
        m = self.meter
        if m is None:
            return
        if m.error:
            err = m.error
            self.toggle_meter()
            self.lbl_meter.configure(text=ellipsize(FONTS["small"], f"Mic error: {err}", 230), fg=C["live_hi"])
            return
        db = m.level
        self.level.set_level(db)
        if db > -40:
            self.lbl_meter.configure(text="Hearing you  ✓", fg=C["ok"])
        elif db > -55:
            self.lbl_meter.configure(text="Very quiet — wrong input or muted?", fg=C["warn"])
        else:
            self.lbl_meter.configure(text="Silent — try another input", fg=C["muted"])

    # ---------------------------------------------------------- engine
    def start_engine(self, live=False):
        if not self.cam_path():
            self.alert("No webcam found", "Plug in or enable a webcam, then press refresh in Setup.")
            return False
        self.engine = Engine(self.collect(), live=live, screen=self.screen)
        self.engine.start()
        return True

    def stop_engine(self):
        self._rec_pending = False
        if self.recorder is not None:
            self.stop_recording()
        if self.engine is not None:
            self.engine.stop()
            self.engine.join(timeout=3)
            self.engine = None
        self._shown_frame = None
        self.render_preview(force=True)

    def engine_running(self):
        return self.engine is not None and self.engine.is_alive()

    def toggle_camera(self):
        if self.engine_running():
            self.stop_engine()
        else:
            self.start_engine()
            self.render_preview(force=True)

    def restart_if_running(self):
        if self.engine_running():
            live = self.engine.live
            self.stop_engine()
            self.start_engine(live=live)
        self.push()

    def toggle_live(self):
        if not self.engine_running():
            self.start_engine(live=True)
        else:
            self.engine.live = not self.engine.live

    # ---------------------------------------------------------- background choices
    def set_mode(self, m):
        self.v_mode.set(m)
        if m == "screen" and not self.screen.active:
            self.screen.start(reuse=True)
        self.show_context()
        self.push()

    def toggle_screen(self):
        if self.screen.active:
            self.screen.stop()
        else:
            self.screen.start(reuse=True)
            self.set_mode("screen")

    def change_screen(self):
        self.screen.start(reuse=False)
        self.set_mode("screen")

    def pick_file(self, title, patterns, directory=False):
        """Native file picker via zenity when available; Tk dialog otherwise."""
        if shutil.which("zenity"):
            cmd = ["zenity", "--file-selection", f"--title={title}"]
            if directory:
                cmd.append("--directory")
            else:
                cmd.append(f"--file-filter={title} | {patterns}")
            try:
                r = subprocess.run(cmd, capture_output=True, text=True)
                return r.stdout.strip() or None
            except Exception:
                pass
        if directory:
            return filedialog.askdirectory(title=title) or None
        return filedialog.askopenfilename(title=title, filetypes=[(title, patterns), ("All", "*.*")]) or None

    def pick_image(self):
        p = self.pick_file("Choose a background image", "*.png *.jpg *.jpeg *.webp *.bmp")
        if p:
            self.v_image.set(p)
            self.set_mode("image")

    def use_default_image(self):
        self.v_image.set("")
        self.update_context()

    def pick_video(self):
        p = self.pick_file("Choose a background video", "*.mp4 *.mkv *.mov *.webm *.avi")
        if p:
            self.v_video.set(p)
            self.set_mode("video")

    def set_colour(self, hx):
        self.v_colour.set(hx)
        self.update_context()

    def pick_colour(self):
        hx = None
        if shutil.which("zenity"):
            try:
                r = subprocess.run(["zenity", "--color-selection", "--title=Background colour",
                                    f"--color={self.v_colour.get()}"], capture_output=True, text=True)
                m = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)", r.stdout.strip())
                if m:
                    hx = "#%02x%02x%02x" % tuple(int(x) for x in m.groups())
                elif r.stdout.strip().startswith("#"):
                    hx = r.stdout.strip()[:7]
            except Exception:
                pass
        else:
            _, hx = colorchooser.askcolor(color=self.v_colour.get(), title="Background colour")
        if hx:
            self.set_colour(hx)

    def reset_cutout(self):
        self.v_soft.set(DEFAULTS["softness"])
        self.v_thr.set(DEFAULTS["threshold"])
        self.v_smooth.set(DEFAULTS["smoothing"])
        self.v_invert.set(False)

    # ---------------------------------------------------------- recording
    def rec_dir(self):
        return self.v_recdir.get() or DEFAULTS["record_dir"]

    def update_recdir_label(self):
        d = self.rec_dir().replace(os.path.expanduser("~"), "~", 1)
        self.lbl_recdir.configure(text=ellipsize(FONTS["small"], d, 330))

    def change_rec_dir(self):
        p = self.pick_file("Choose where recordings are saved", "", directory=True)
        if p:
            self.v_recdir.set(p)
            self.update_recdir_label()

    def toggle_record(self):
        if self.recorder is not None or self._rec_pending:
            self._rec_pending = False
            self.stop_recording()
            return
        if shutil.which("ffmpeg") is None:
            self.alert("ffmpeg is missing", "Recording needs ffmpeg.\n\nRun:  sudo apt install ffmpeg")
            return
        if not self.engine_running():
            if not self.start_engine():
                return
        self._rec_pending = True
        self.btn_rec.set(text="Starting…")

    def begin_recording(self):
        w, h = self.engine.size
        d = self.rec_dir()
        try:
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, time.strftime("NexusCam_%Y-%m-%d_%H-%M-%S.mp4"))
            self.recorder = Recorder(w, h, DEFAULTS["fps"], path, audio=bool(self.v_mic.get()),
                                     source=self.mic_name())
        except Exception as e:
            self.recorder = None
            self.alert("Couldn't start recording", str(e))
            return
        self.engine.recorder = self.recorder

    def stop_recording(self):
        rec, self.recorder = self.recorder, None
        if self.engine is not None:
            self.engine.recorder = None
        if rec is None:
            return
        self.btn_rec.set(text="Saving…", icon_name=None, enabled=False)
        self.root.update_idletasks()
        err = rec.stop()
        self.btn_rec.set(enabled=True)
        if err:
            if self.v_mic.get():
                err += "\n\nIf this is a microphone problem, pick another input in the Audio tab."
            self.alert("Recording failed", err)
        else:
            self._last_saved = rec.path
            mins, secs = divmod(int(rec.elapsed), 60)
            self.toast(f"Saved {mins}:{secs:02d} recording", action=("Open folder", self.open_rec_folder))
            if rec.warning:
                self.alert("No sound in the recording", rec.warning, kind="warn")

    def open_rec_folder(self):
        d = self.rec_dir()
        os.makedirs(d, exist_ok=True)
        try:
            subprocess.Popen(["xdg-open", d], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            self.alert("Recordings folder", d, kind="info")

    # ---------------------------------------------------------- feedback
    def alert(self, title, message, kind="error"):
        top = tk.Toplevel(self.root)
        top.withdraw()
        top.title("NexusCam")
        top.configure(bg=C["raised"])
        top.transient(self.root)
        top.resizable(False, False)
        body = tk.Frame(top, bg=C["raised"])
        body.pack(padx=26, pady=24)
        r = tk.Frame(body, bg=C["raised"])
        r.pack(fill="x")
        colour = {"error": C["live"], "warn": C["warn"]}.get(kind, C["accent_hi"])
        top._ic = icon("alert" if kind != "info" else "info", 30, colour, C["raised"])
        tk.Label(r, image=top._ic, bg=C["raised"]).pack(side="left", anchor="n")
        txt = tk.Frame(r, bg=C["raised"])
        txt.pack(side="left", padx=(16, 0))
        tk.Label(txt, text=title, font=FONTS["big"], fg=C["text"], bg=C["raised"]).pack(anchor="w")
        for prefix in ("Recording failed.", "Recording failed:", "Screen capture stopped:"):
            if message.startswith(prefix):
                message = message[len(prefix):].strip()
        tk.Label(txt, text=message, font=FONTS["body"], fg=C["muted"], bg=C["raised"],
                 wraplength=440, justify="left").pack(anchor="w", pady=(8, 0))
        btns = tk.Frame(body, bg=C["raised"])
        btns.pack(fill="x", pady=(22, 0))
        Btn(btns, "OK", None, "primary", top.destroy, bg=C["raised"], width=100, height=38).pack(side="right")
        top.bind("<Return>", lambda e: top.destroy())
        top.bind("<Escape>", lambda e: top.destroy())
        top.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - top.winfo_reqwidth()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - top.winfo_reqheight()) // 3
        top.geometry(f"+{max(0, x)}+{max(0, y)}")
        top.deiconify()
        try:
            top.grab_set()
        except tk.TclError:
            pass
        top.focus_set()

    def toast(self, text, kind="info", action=None, ms=5000):
        """Floating notification at the bottom of the preview."""
        if self._toast_after:
            self.root.after_cancel(self._toast_after)
        self._toast = (text, kind, action)
        self.render_toast()
        self._toast_after = self.root.after(ms, self.clear_toast)

    def clear_toast(self):
        self._toast = None
        self._toast_after = None
        self.preview.delete("toast")

    def render_toast(self):
        cv = self.preview
        cv.delete("toast")
        t = getattr(self, "_toast", None)
        if not t:
            return
        text, kind, action = t
        dot = {"info": C["accent_hi"], "ok": C["ok"], "warn": C["warn"], "error": C["live"]}.get(kind, C["accent_hi"])
        if kind == "info" and action:
            dot = C["ok"]
        img = pill_rgba(text, C["raised"], C["text"], dot=dot, alpha=245, size=14, padx=16, h=40)
        self._toast_img = ImageTk.PhotoImage(img)
        W, H = cv.winfo_width(), cv.winfo_height()
        aw = (FONTS["bold"].measure(action[0]) + 26) if action else 0
        total = img.width + (aw + 8 if action else 0)
        x0 = (W - total) // 2
        vr = getattr(self, "_video_rect", None)
        y = (vr[3] - 20) if (vr and self._shown_frame is not None) else H - 28
        cv.create_image(x0, y, image=self._toast_img, anchor="sw", tags="toast")
        if action:
            ax = x0 + img.width + 8
            self._toast_btn = ImageTk.PhotoImage(pill_rgba(action[0], C["accent"], "#ffffff",
                                                           alpha=255, size=14, padx=13, h=40))
            item = cv.create_image(ax, y, image=self._toast_btn, anchor="sw", tags=("toast", "toastbtn"))
            cv.tag_bind("toastbtn", "<ButtonRelease-1>", lambda e: (action[1](), self.clear_toast()))
            cv.tag_bind("toastbtn", "<Enter>", lambda e: cv.configure(cursor="hand2"))
            cv.tag_bind("toastbtn", "<Leave>", lambda e: cv.configure(cursor=""))

    # ---------------------------------------------------------- preview
    def render_preview(self, force=False):
        cv = self.preview
        W, H = cv.winfo_width(), cv.winfo_height()
        if W < 20 or H < 20:
            return
        e = self.engine
        frame = e.frame_out if (e is not None and e.is_alive()) else None
        if frame is None:
            starting = e is not None and e.is_alive()
            if force or self._shown_frame is not None or getattr(self, "_empty_state", None) != starting:
                self._shown_frame = None
                self._empty_state = starting
                self.draw_empty(W, H, starting=starting)
            return
        self._empty_state = None
        if frame is self._shown_frame and not force:
            return
        self._shown_frame = frame
        fh, fw = frame.shape[:2]
        sc = min(W / fw, H / fh)
        pw, ph = max(1, int(fw * sc)), max(1, int(fh * sc))
        small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(small)
        mask = self._masks.get((pw, ph))
        if mask is None:
            S = 3
            m = Image.new("L", (pw * S, ph * S), 0)
            ImageDraw.Draw(m).rounded_rectangle([0, 0, pw * S - 1, ph * S - 1], radius=14 * S, fill=255)
            mask = m.resize((pw, ph), Image.LANCZOS)
            self._masks = {(pw, ph): mask}
        base = Image.new("RGB", (pw, ph), _rgb(C["bg"]))
        base.paste(pil, (0, 0), mask)
        # badges
        x = 16
        if e.vcam_open:
            b = self._badge("LIVE", C["live"], "#ffffff", "#ffffff")
            base.paste(b, (x, 16), b)
            x += b.width + 8
        if self.recorder is not None:
            mins, secs = divmod(int(self.recorder.elapsed), 60)
            b = self._badge(f"REC  {mins:02d}:{secs:02d}", "#0a0e13", "#ffffff", C["live"])
            base.paste(b, (x, 16), b)
        self._photo = ImageTk.PhotoImage(base)
        self._video_rect = (W // 2 - pw // 2, H // 2 - ph // 2, W // 2 + pw // 2, H // 2 + ph // 2)
        cv.delete("empty")
        if not cv.find_withtag("video"):
            cv.create_image(W // 2, H // 2, image=self._photo, tags="video")
        else:
            cv.coords("video", W // 2, H // 2)
            cv.itemconfigure("video", image=self._photo)
        cv.tag_raise("toast")

    def _badge(self, text, fill, fg, dot):
        key = (text, fill)
        b = self._badges.get(key)
        if b is None:
            if len(self._badges) > 50:
                self._badges = {}
            b = pill_rgba(text, fill, fg, dot=dot, alpha=230 if fill != C["live"] else 255, size=13)
            self._badges[key] = b
        return b

    def draw_empty(self, W, H, starting=False):
        cv = self.preview
        cv.delete("video")
        cv.delete("empty")
        self._empty_bg = rr_image(W, H, 16, C["panel"], C["bg"], C["line"])
        cv.create_image(0, 0, image=self._empty_bg, anchor="nw", tags="empty")
        cy = H // 2
        self._empty_ic = icon("camera", 44, C["faint"], C["panel"])
        cv.create_image(W // 2, cy - 70, image=self._empty_ic, tags="empty")
        cv.create_text(W // 2, cy - 22, text="Starting camera…" if starting else "Camera is off",
                       font=FONTS["hero"], fill=C["text"], tags="empty")
        sub = ("Loading the cut-out model — this takes a moment the first time."
               if starting else "Choose your webcam in Setup, then start it. Your scene appears here.")
        cv.create_text(W // 2, cy + 8, text=sub, font=FONTS["body"], fill=C["muted"], tags="empty")
        if not starting:
            cv.create_window(W // 2, cy + 60, window=self.start_overlay, tags="empty")
        cv.tag_raise("toast")

    # ---------------------------------------------------------- keys / icons
    def bind_keys(self):
        def handler(fn):
            def h(e):
                if isinstance(e.widget, tk.Entry):
                    return
                fn()
            return h
        for i, m in enumerate(MODES, start=1):
            self.root.bind(f"<Key-{i}>", handler(lambda m=m: self.set_mode(m)))
        self.root.bind("<space>", handler(self.toggle_live))
        self.root.bind("<Key-r>", handler(self.toggle_record))
        self.root.bind("<Key-R>", handler(self.toggle_record))

    def set_icons(self):
        try:
            big = logo_pil(256)
            path = os.path.join(APP_DIR, "icon.png")
            big.save(path)
            self._wicon = ImageTk.PhotoImage(logo_pil(64))
            self.root.iconphoto(True, self._wicon)
            desk = os.path.join(os.path.expanduser("~"), ".local/share/applications/nexuscam.desktop")
            if os.path.exists(desk):
                txt = open(desk).read()
                if "Icon=camera-web" in txt:
                    with open(desk, "w") as f:
                        f.write(txt.replace("Icon=camera-web", f"Icon={path}"))
        except Exception:
            pass

    # ---------------------------------------------------------- UI loop
    def tick(self):
        if self.screen.error:
            msg, self.screen.error = self.screen.error, None
            self.alert("Screen sharing stopped", msg)
        self.draw_meter()

        # screen status (Scene tab)
        st = self.screen.status
        if st != self._last_screen_status:
            self._last_screen_status = st
            colour = (C["ok"] if st == "Sharing" else C["warn"] if self.screen.active else C["faint"])
            self.scr_dot.configure(fg=colour)
            self.scr_status.configure(text=st)
            if self.screen.active:
                self.btn_share.set(text="Stop sharing", icon_name="stop", variant="secondary")
            else:
                self.btn_share.set(text="Share a window", icon_name="screen", variant="primary")

        e = self.engine
        running = e is not None and e.is_alive()
        if e is not None:
            if e.error:
                msg, e.error = e.error, None
                self.alert("Camera problem", msg)
            if not running and not e.running:
                self.stop_engine()
                e, running = None, False
        if running:
            if self._rec_pending and e.size != (0, 0):
                self._rec_pending = False
                self.begin_recording()
            if self.recorder is not None and not self.recorder.alive:
                self.stop_recording()

        live = running and e.vcam_open
        if live != self._was_live:
            self._was_live = live
            self.toast(f"You're live on {self.v_vdev.get()}" if live else "Live ended",
                       kind="ok" if live else "info", ms=3000)

        # transport buttons
        self.btn_cam.set(text="Stop camera" if running else "Start camera",
                         icon_name="stop" if running else "camera")
        self.btn_live.set(text="End live" if live else "Go live",
                          variant="danger" if live else "primary")
        if self.recorder is not None:
            mins, secs = divmod(int(self.recorder.elapsed), 60)
            self.btn_rec.set(text=f"Stop  {mins:02d}:{secs:02d}", icon_name="stop", variant="danger")
        elif not self._rec_pending and self.btn_rec.enabled:
            self.btn_rec.set(text="Record", icon_name="rec", variant="danger_soft")

        # header chips
        self.chip_cam.set("Camera on" if running else "Camera off", "on" if running else "off")
        self.chip_live.set("LIVE", "live")
        self.chip_live.show(live)
        if self.recorder is not None:
            mins, secs = divmod(int(self.recorder.elapsed), 60)
            self.chip_rec.set(f"REC {mins:02d}:{secs:02d}", "rec")
        self.chip_rec.show(self.recorder is not None)

        # status bar
        if running and e.size != (0, 0):
            w, h = e.size
            self.lbl_stats.configure(
                text=f"{w}×{h}   ·   {e.fps:.0f} fps   ·   {e.proc_ms:.0f} ms per frame   ·   "
                     f"{MODE_LABELS[self.v_mode.get()]} background")
        else:
            self.lbl_stats.configure(text="Camera off" if not running else "Starting camera…")
        self.lbl_enc.configure(text=f"Encoder: {self.recorder.encoder}" if self.recorder else "")

        self.render_preview()
        self.root.after(15, self.tick)

    def on_close(self):
        self.save_settings()
        if self.meter is not None:
            self.meter.stop()
        self.stop_engine()
        self.screen.stop()
        self.root.destroy()


def main():
    root = tk.Tk(className="NexusCam")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

import atexit
import functools
import http.server
import os
import socket
import tempfile
import threading

import cv2
import streamlit as st


_active_servers: dict[str, dict] = {}  # label -> {"server": HTTPServer, "path": str}
_rendered_paths: list[str] = []        # all temp files created this session


def _cleanup_temp_files() -> None:
    for p in _rendered_paths:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass

atexit.register(_cleanup_temp_files)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start_video_server(video_path: str, label: str = "player") -> str:
    """Shut down any previous server for this label, then serve video_path on a new port."""
    global _active_servers
    if label in _active_servers:
        _active_servers[label]["server"].shutdown()
        _active_servers.pop(label)

    directory = os.path.dirname(video_path)
    filename  = os.path.basename(video_path)
    port      = _free_port()

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            """Serve the file with HTTP range-request support so browsers can seek."""
            path = self.translate_path(self.path)
            try:
                file_size = os.path.getsize(path)
                f = open(path, "rb")
            except (OSError, FileNotFoundError):
                self.send_error(404)
                return

            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                # Parse the first byte range (e.g. "bytes=0-1023")
                ranges = range_header[6:].split(",")[0].strip()
                start_str, _, end_str = ranges.partition("-")
                start = int(start_str) if start_str else 0
                end   = int(end_str)   if end_str   else file_size - 1
                end   = min(end, file_size - 1)
                length = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                f.seek(start)
                remaining = length
                while remaining:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    remaining -= len(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(file_size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
            f.close()

    class _QuietServer(http.server.HTTPServer):
        def handle_error(self, request, client_address):
            pass  # suppress connection-reset noise from browser early-close

    handler = functools.partial(_QuietHandler, directory=directory)
    server  = _QuietServer(("localhost", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://localhost:{port}/{filename}"
    _active_servers[label] = {"server": server, "path": video_path}
    return url


def render_video(frames, fps, label="Rendering video") -> str | None:
    """Write RGB frames to a persistent temp file (H.264). Returns the file path."""
    h, w = frames[0].shape[:2]

    out_tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = out_tmp.name
    out_tmp.close()
    _rendered_paths.append(out_path)

    fourcc = cv2.VideoWriter.fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        st.error("H.264 (avc1) codec unavailable. Install ffmpeg and add it to PATH.")
        os.unlink(out_path)
        return None

    progress = st.progress(0, text=label)
    for i, frame_rgb in enumerate(frames):
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        progress.progress((i + 1) / len(frames),
                          text=f"{label} — frame {i+1}/{len(frames)}")
    writer.release()
    progress.empty()
    return out_path
"""Download a recording from a link (YouTube, Google Drive, Zoom cloud recording, direct URL, ...).

Uses yt-dlp, which covers all of these. The file lands in the caller's temp dir,
so the pipeline's cleanup deletes it with everything else.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from .config import get_config
from .media import MediaError

log = logging.getLogger(__name__)
ProgressFn = Callable[[float], None]


def check_url(url: str) -> str:
    """Allow only public http(s) URLs, so a deployed app can't be pointed at internal hosts."""
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise MediaError("Please paste a full http(s) link to the recording.")
    try:
        addrs = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, None)}
    except socket.gaierror:
        raise MediaError("That link's website could not be found. Check the URL.")
    for addr in addrs:
        ip = ipaddress.ip_address(addr.split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise MediaError("Links to private or local network addresses are not allowed.")
    return url


def download(url: str, out_dir: str | Path, on_progress: ProgressFn | None = None) -> Path:
    import yt_dlp  # imported lazily: only needed for link input

    cfg = get_config()
    url = check_url(url)
    max_bytes = cfg["media"]["max_upload_mb"] * 1_000_000
    max_dur = cfg["media"]["max_duration_s"]

    def hook(d: dict) -> None:
        if on_progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                on_progress(min(d.get("downloaded_bytes", 0) / total, 1.0))

    def too_long(info: dict, *, incomplete: bool) -> str | None:
        dur = info.get("duration")
        return f"recording longer than {max_dur // 60} min" if dur and dur > max_dur else None

    opts = {
        # 720p is plenty for frame checks and keeps downloads small; audio is always taken at best quality.
        "format": "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(Path(out_dir) / "download.%(ext)s"),
        "noplaylist": True,
        "max_filesize": max_bytes,
        "match_filter": too_long,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        "logger": _SilentLogger(),  # yt-dlp's own messages echo URLs/titles; we report errors ourselves
        "ffmpeg_location": _ffmpeg_dir(),
    }
    info, last_error = None, ""
    # YouTube often blocks cloud servers with a bot check for its default web player but still serves other
    # player types, so on a bot check we retry as those before giving up.
    for clients in (None, ["tv", "mweb"], ["web_safari", "ios"]):
        attempt = dict(opts)
        if clients:
            attempt["extractor_args"] = {"youtube": {"player_client": clients}}
        try:
            with yt_dlp.YoutubeDL(attempt) as ydl:
                info = ydl.extract_info(url, download=True)
            last_error = ""
            break
        except yt_dlp.utils.DownloadError as e:
            last_error = str(e)
            if not _is_bot_check(last_error):
                break
            log.warning("link blocked by a bot check; retrying with another player type")
    if last_error:
        raise MediaError(_friendly(last_error)) from None

    if info is None:  # rejected by match_filter
        raise MediaError(f"The recording is longer than the {max_dur // 60}-minute limit.")
    files = [p for p in Path(out_dir).glob("download.*") if p.suffix not in (".part", ".ytdl")]
    if not files:
        raise MediaError("The link did not produce a downloadable video (it may be over the size limit).")
    log.info("downloaded recording from %s", urlparse(url).hostname)  # host only, never the full URL
    return files[0]


class _SilentLogger:
    def debug(self, msg: str) -> None: ...
    def info(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...


def _ffmpeg_dir() -> str | None:
    import shutil

    exe = shutil.which(get_config()["ffmpeg_bin"])
    return str(Path(exe).parent) if exe else None


def _is_bot_check(err: str) -> bool:
    low = err.lower()
    return "not a bot" in low or "confirm you" in low or ("sign in" in low and "youtube" in low)


def _friendly(err: str) -> str:
    low = err.lower()
    if _is_bot_check(err):
        return ("YouTube did not allow this server to download the video (it blocks many cloud servers). Please "
                "download the video and upload the file instead.")
    if any(k in low for k in ("sign in", "login", "private", "permission", "403", "401", "access denied")):
        return ("The link is private. Share it as 'Anyone with the link can view' (Google Drive / Zoom), "
                "or download the recording and upload the file instead.")
    if "passcode" in low or "password" in low:
        return "This recording is passcode-protected. Remove the passcode or upload the file instead."
    if "unsupported url" in low:
        return "This link isn't a supported video page. Try a direct link, or upload the file."
    if "file is larger" in low or "max_filesize" in low:
        return f"The recording is over the {get_config()['media']['max_upload_mb']} MB limit."
    return "The recording could not be downloaded from that link. Check it opens in a browser, or upload the file."

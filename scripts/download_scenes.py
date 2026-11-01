"""
Download the example scenes used in the paper.

Each scene config in conf/scene/ has a `source` block naming an archive in the
WISION Lab public S3 bucket and the recording inside it. The frame window a
scene uses is set at load time by its config.

- Zip archives hold many recordings (~100 GB each), so instead of downloading
  them we read the zip index with HTTP range requests and fetch only that
  recording (2-3 GB each).
- Tar archives (the UbiCam night-driving set, 3.7 GB compressed) can't be
  range-read; they are streamed once and only the needed files are written out
  (the recordings plus any `extra` files, such as the sensor's hot-pixel mask).

    data/
      scenes/<scene>.npy                 full bit-packed photon cube, (t, h, w/8) uint8
      scenes/<scene>.h5                  UbiCam recording (pre-summed counts)
      hot_pixel_mask/*.npy               sensor hot-pixel masks
      color_filter_array/*.tif           RGBW colour filter layout (colour scenes)

Usage:
    uv run python scripts/download_scenes.py                 # all scenes
    uv run python scripts/download_scenes.py newton_cradle   # some scenes
    uv run python scripts/download_scenes.py --list
"""

import argparse
import io
import struct
import sys
import tarfile
import urllib.request
from collections import defaultdict
import zipfile
from pathlib import Path

import yaml
from tqdm import tqdm

S3_BASE_URL = "https://web.s3.wisc.edu/public-datasets/"
MASKS_ARCHIVE = "quanta-vision/sequences/masks.zip"
MASK_DIRS = ("masks/hot_pixel_mask/", "masks/color_filter_array/")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_DIR = REPO_ROOT / "conf" / "scene"
DATA_DIR = REPO_ROOT / "data"


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer):
        data = self.read_range(self.pos, len(buffer))
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)

    def read_range(self, start: int, length: int) -> bytes:
        end = min(start + length, self.size) - 1
        if end < start:
            return b""
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request) as response:
            return response.read()

    def stream_range(self, start: int, length: int, out, chunk: int = 1 << 24):
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{start + length - 1}"}
        )
        with urllib.request.urlopen(request) as response, tqdm(
            total=length, unit="B", unit_scale=True, leave=False
        ) as bar:
            while block := response.read(chunk):
                out.write(block)
                bar.update(len(block))


def open_archive(key: str) -> tuple[HTTPRangeFile, zipfile.ZipFile]:
    remote = HTTPRangeFile(S3_BASE_URL + key)
    return remote, zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20))


def member_data_offset(remote: HTTPRangeFile, info: zipfile.ZipInfo) -> int:
    """Byte offset of a stored member's data within the archive."""
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{info.filename} is compressed; cannot range-read it")
    header = remote.read_range(info.header_offset, 30)
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    return info.header_offset + 30 + name_len + extra_len


def load_scenes() -> dict[str, dict]:
    scenes = {}
    for path in sorted(SCENE_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        if cfg and "source" in cfg:
            scenes[path.stem] = cfg["source"]
    return scenes


def scene_path(name: str, source: dict) -> Path:
    return DATA_DIR / "scenes" / f"{name}{Path(source['member']).suffix}"


def is_tar(source: dict) -> bool:
    return source["archive"].endswith((".tar", ".tar.gz", ".tgz", ".tar.xz"))


def download_tar(archive: str, wanted: dict[str, Path], force: bool = False):
    """Stream a tar archive once, writing the members in `wanted` to their paths."""
    todo = {m: p for m, p in wanted.items() if force or not p.exists()}
    for member, path in wanted.items():
        if member not in todo:
            print(f"[skip] {path.relative_to(REPO_ROOT)} exists")
    if not todo:
        return
    print(f"[get]  streaming {archive} for {len(todo)} file(s)")
    with urllib.request.urlopen(S3_BASE_URL + archive) as response:
        total = int(response.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, leave=False) as bar:
            reader = _ProgressReader(response, bar)
            with tarfile.open(fileobj=reader, mode="r|*") as tar:
                for info in tar:
                    if info.name not in todo or not info.isfile():
                        continue
                    path = todo.pop(info.name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = path.with_name(path.name + ".part")
                    with tar.extractfile(info) as src, open(tmp_path, "wb") as out:
                        while block := src.read(1 << 24):
                            out.write(block)
                    tmp_path.rename(path)
                    print(f"[ok]   {path.relative_to(REPO_ROOT)} ({info.size / 1e9:.1f} GB)")
                    if not todo:
                        break
    if todo:
        raise RuntimeError(f"{archive} has no member(s) {sorted(todo)}")


class _ProgressReader(io.RawIOBase):
    def __init__(self, raw, bar):
        self.raw, self.bar = raw, bar

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.raw.read(len(buffer))
        buffer[: len(data)] = data
        self.bar.update(len(data))
        return len(data)


def download_scene(name: str, source: dict, force: bool = False):
    out_path = scene_path(name, source)
    if out_path.exists() and not force:
        print(f"[skip] {name}: {out_path.relative_to(REPO_ROOT)} exists")
        return

    remote, archive = open_archive(source["archive"])
    info = archive.getinfo(source["member"])
    data_start = member_data_offset(remote, info)

    print(f"[get]  {name}: {source['member']} ({info.file_size / 1e9:.1f} GB)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".part")
    with open(tmp_path, "wb") as out:
        remote.stream_range(data_start, info.file_size, out)
    tmp_path.rename(out_path)


def download_masks(force: bool = False):
    remote, archive = open_archive(MASKS_ARCHIVE)
    for info in archive.infolist():
        if info.is_dir() or not info.filename.startswith(MASK_DIRS):
            continue
        out_path = DATA_DIR / info.filename.removeprefix("masks/")
        if out_path.exists() and not force:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(archive.read(info))
    print(f"[ok]   masks in {DATA_DIR.relative_to(REPO_ROOT)}/")


def main():
    scenes = load_scenes()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scenes", nargs="*", help=f"default: all of {sorted(scenes)}")
    parser.add_argument("--list", action="store_true", help="list scenes and exit")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    if args.list:
        for name, src in scenes.items():
            print(f"{name:24s} {src['member']}")
        return

    unknown = set(args.scenes) - set(scenes)
    if unknown:
        sys.exit(f"Unknown scene(s): {sorted(unknown)}. Use --list.")

    download_masks(args.force)
    tars: dict[str, dict[str, Path]] = defaultdict(dict)
    for name in args.scenes or scenes:
        source = scenes[name]
        if is_tar(source):
            tars[source["archive"]][source["member"]] = scene_path(name, source)
            for member, path in source.get("extra", {}).items():
                tars[source["archive"]][member] = REPO_ROOT / path
        else:
            download_scene(name, source, args.force)
    for archive, wanted in tars.items():
        download_tar(archive, wanted, args.force)


if __name__ == "__main__":
    main()

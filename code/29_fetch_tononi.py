"""
Download Tononi Serial Awakenings subject archives, with verification.

A first attempt used a plain curl loop and silently produced truncated files:
three of four archives were 18-27% of their true size and unreadable as zips.
Truncated EEG would not raise an error downstream, it would produce wrong
numbers. Every download here is therefore checked twice -- byte size against the
figshare manifest, and zip-integrity via testzip() -- and retried on failure.

Usage:  python 29_fetch_tononi.py [n_subjects]
Re-running is safe: complete and valid archives are skipped.
"""
import os
import sys
import json
import time
import zipfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "tononi")
API = "https://api.figshare.com/v2/articles/23306054"
MAX_TRIES = 4
CHUNK = 1 << 20

os.makedirs(DEST, exist_ok=True)


def manifest():
    """Subject archives only.

    Part1 is named '...-Part1-No_PSGs.zip' and contains the reports and montage,
    not EEG. A naive '_PSGs' match includes it, so it is excluded explicitly.
    """
    with urllib.request.urlopen(API, timeout=60) as r:
        d = json.load(r)
    fs = [f for f in d["files"]
          if "_PSGs" in f["name"] and "No_PSGs" not in f["name"]]
    fs.sort(key=lambda f: int(f["name"].split("Part")[1].split("-")[0]))
    return fs


def local_name(fname):
    tag = fname.split("-")[-2] + "-" + fname.split("-")[-1].replace("_PSGs.zip", "")
    return os.path.join(DEST, tag + ".zip")


def verify(path, expected):
    """Size must match exactly and the archive must open cleanly."""
    if not os.path.exists(path):
        return False, "absent"
    got = os.path.getsize(path)
    if got != expected:
        return False, f"size {got/1e6:.1f}MB != {expected/1e6:.1f}MB"
    try:
        if zipfile.ZipFile(path).testzip() is not None:
            return False, "corrupt member"
    except Exception as e:
        return False, f"unreadable ({type(e).__name__})"
    return True, "ok"


def fetch(url, path, expected):
    """Resume-capable download; returns True only if verification passes."""
    for attempt in range(1, MAX_TRIES + 1):
        ok, why = verify(path, expected)
        if ok:
            return True
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have > expected:          # previous partial is unusable
            os.remove(path)
            have = 0
        req = urllib.request.Request(url)
        mode = "wb"
        if have:
            req.add_header("Range", f"bytes={have}-")
            mode = "ab"
        try:
            with urllib.request.urlopen(req, timeout=120) as r, \
                    open(path, mode) as fh:
                while True:
                    b = r.read(CHUNK)
                    if not b:
                        break
                    fh.write(b)
        except Exception as e:
            print(f"      attempt {attempt} failed: {type(e).__name__}")
            time.sleep(3 * attempt)
            continue
        ok, why = verify(path, expected)
        if ok:
            return True
        print(f"      attempt {attempt}: {why}")
        time.sleep(2 * attempt)
    return False


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    fs = manifest()[:n]
    print(f"target: {len(fs)} subject archives, "
          f"{sum(f['size'] for f in fs)/1e9:.2f} GB total\n")

    good, bad = [], []
    for i, f in enumerate(fs, 1):
        path = local_name(f["name"])
        tag = os.path.basename(path)
        ok, why = verify(path, f["size"])
        if ok:
            print(f"[{i:>2}/{len(fs)}] {tag:<18} already complete")
            good.append(tag)
            continue
        print(f"[{i:>2}/{len(fs)}] {tag:<18} {f['size']/1e6:>6.1f} MB  ({why})")
        if fetch(f["download_url"], path, f["size"]):
            print(f"          verified")
            good.append(tag)
        else:
            print(f"          FAILED after {MAX_TRIES} attempts")
            bad.append(tag)

    print(f"\nverified {len(good)}/{len(fs)}")
    if bad:
        print("failed:", bad)
        print("re-run this script to retry; completed files are skipped")


if __name__ == "__main__":
    main()

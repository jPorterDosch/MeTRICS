"""
Download ScanNet++ data

Default: download splits with scene IDs and default files
that can be used for novel view synthesis on DSLR and iPhone images
and semantic tasks on the mesh
"""

import argparse
import time
from pathlib import Path
import urllib.request
from urllib.request import urlretrieve
import urllib.error
import yaml
from munch import Munch
from tqdm import tqdm
import json
import os
import re
import sys
import zipfile

# vendored verbatim from github.com/scannetpp/scannetpp (common/scene_release.py),
# which is public and MIT-licensed; verified byte-identical to
# raw.githubusercontent.com/scannetpp/scannetpp/main/common/scene_release.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_release import ScannetppScene_Release  # noqa: E402


# --- pretty, colorized terminal error messages ---------------------------
class _Ansi:
    RED = "\033[31m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def _use_color():
    # Only emit ANSI when writing to a real terminal that wants it.
    return (
        sys.stdout.isatty()
        and os.environ.get("NO_COLOR") is None
        and os.environ.get("TERM", "") != "dumb"
    )


def _c(code, text):
    return f"{code}{text}{_Ansi.RESET}" if _use_color() else text


def _highlight_urls(text):
    # Make any http(s) URL stand out so it's obvious where to go. Trailing
    # punctuation (comma, period, paren, quote) is kept out of the link.
    return re.sub(
        r"(https?://[^\s]+?)([.,;:)\]'\"]*)(?=\s|$)",
        lambda m: _c(_Ansi.CYAN + _Ansi.UNDERLINE, m.group(1)) + m.group(2),
        text,
    )


def print_error_box(title, body, accent=_Ansi.RED):
    """Print a framed, colored error block so failures (and what to do about
    them) are easy to spot in a long download log."""
    bar = "─" * 72
    print()
    print(_c(accent + _Ansi.BOLD, f"✗  {title}"))
    print(_c(accent, bar))
    print(_highlight_urls(body))
    print(_c(accent, bar))
    print()


def print_banner(config_path):
    """Print the ScanNet++ ASCII logo from the top of the config file."""
    try:
        with open(config_path) as f:
            lines = f.read().splitlines()
    except Exception:
        return
    if not lines or not lines[0].startswith("#"):
        return
    # Grab the first '#'-bordered box at the very top (the logo), stopping at
    # its closing border (the second all-'#' row).
    banner = [lines[0]]
    for ln in lines[1:]:
        if not ln.startswith("#"):
            break
        banner.append(ln)
        if set(ln.strip()) == {"#"} and len(ln.strip()) > 20:
            break
    print(_c(_Ansi.CYAN + _Ansi.BOLD, "\n".join(banner)))
    print()


def read_txt_list(path):
    with open(path) as f:
        lines = f.read().splitlines()

    return lines


def load_json(path):
    with open(path) as f:
        j = json.load(f)

    return j


def load_yaml_munch(path):
    with open(path) as f:
        y = yaml.load(f, Loader=yaml.Loader)

    return Munch.fromDict(y)


def check_remote_file_exists(url):
    """
    Checks that a given URL is reachable.
    :param url: A URL
    :rtype: bool
    """
    request = urllib.request.Request(url)
    request.get_method = lambda: "HEAD"

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


def download_scannetpp_gs(cfg, scene_ids):
    '''
    Download ScanNet++GS data
    '''

    print('Downloading ScanNet++GS data...')
    for scene_id in tqdm(scene_ids, desc="scenes"):
        src_path = Path('scannetpp_gs') / scene_id / 'ckpts' / 'point_cloud_30000.ply'
        tgt_path = Path(cfg.scannetpp_gs_dir) / f"{scene_id}/point_cloud_30000.ply"
        if not check_download_file(cfg, cfg.scannetpp_gs_url, src_path, tgt_path, cfg.dry_run):
            break


def urlretrieve_multi_trials(url, filename, max_trials=5):
    # 401 -> invalid token or token expired
    # 404 -> File not found
    # 406 -> Not acceptable, please update the download script
    for i in range(max_trials):
        try:
            urlretrieve(url, filename)
            time.sleep(0.2)     # wait a bit to prevent being rejected for frequent access to the server
            return True

        except urllib.error.ContentTooShortError as e:
            # Failed to download the file completely. Delete the file and retry.
            # Delete filename
            print("ERROR: Content too short. It is likely that the download was incomplete due to network issues. Retrying...")
            if i < max_trials - 1:
                if Path(filename).exists():
                    os.remove(filename)
                time.sleep(0.5)  # wait a bit before retrying
            else:
                print(f"Failed to download {url} after {max_trials} trials")
                raise e

        # HTTP error
        except urllib.error.HTTPError as e:
            if e.code == 401:
                # The server's response body explains whether the token is
                # invalid or expired, and (if expired) how to request an extension.
                try:
                    server_msg = e.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    server_msg = ""
                print_error_box(
                    "Download failed — invalid or expired token",
                    server_msg or "It could be that an invalid or expired token is used.",
                )
                # Stop cleanly: SystemExit prints no traceback, so the message
                # above stays visible instead of being buried under a stack trace.
                raise SystemExit(1)

            elif e.code == 404:
                print_error_box(
                    "Download failed — file not found (404)",
                    f"{url}\n\n"
                    "This asset may not be available for this scene/split.",
                    accent=_Ansi.YELLOW,
                )
                raise SystemExit(1)

            elif e.code == 406:
                try:
                    server_msg = e.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    server_msg = ""
                print_error_box(
                    "Download failed — download script out of date (406)",
                    (server_msg or "This download script is out of date.")
                    + "\n\nPlease download the latest script from the ScanNet++ "
                    "website and try again.",
                    accent=_Ansi.YELLOW,
                )
                raise SystemExit(1)
            elif e.code == 429 or 500 <= e.code < 600:
                # Transient server-side errors. Examples seen in the wild:
                # nginx 500 on NFS-backed small files (sporadic, ~7%) and
                # 429 from the per-IP rate limiter when many files land in
                # the same second. Back off and retry rather than abort
                # the whole download for a single bad request.
                if i < max_trials - 1:
                    delay = (0.5 * (2 ** i))   # 0.5s, 1s, 2s, 4s, 8s
                    print(f"WARN: HTTP {e.code} on {url} — retry {i+1}/{max_trials-1} in {delay:.1f}s")
                    if Path(filename).exists():
                        os.remove(filename)
                    time.sleep(delay)
                    continue
                # Fall through: out of retries.
                try:
                    server_msg = e.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    server_msg = ""
                print_error_box(
                    f"Download failed — server returned HTTP {e.code} after {max_trials} tries",
                    (server_msg + "\n\n" if server_msg else "") + f"URL: {url}",
                )
                raise e
            else:
                try:
                    server_msg = e.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    server_msg = ""
                print_error_box(
                    f"Download failed — server returned HTTP {e.code}",
                    (server_msg + "\n\n" if server_msg else "") + f"URL: {url}",
                )
                raise e
    return False


def zip_is_complete(path):
    """True if `path` is a zip whose central directory reads back cleanly.

    Cheap enough to call per scene per asset: opening a ZipFile parses only
    the central directory at the tail of the file, so this catches the
    truncation a killed job leaves behind without decompressing anything.
    """
    if not Path(path).is_file():
        return False
    try:
        with zipfile.ZipFile(path):
            return True
    except (zipfile.BadZipFile, OSError):
        return False


def download_file(url, filename, verbose=True, make_parent=False):
    """
    Download file from url to filename
    """
    # download_url = url.

    if make_parent:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"{url} ==> {filename}")

    # Download to a sidecar and rename only once the transfer returns.
    # urlretrieve writes straight to the destination, so a job killed
    # mid-transfer (Slurm walltime, node failure) leaves a truncated file
    # under the FINAL name -- which check_download_file() then skips as
    # "File exists" on the next run, baking the corruption in permanently.
    # With the rename, a final-named file is always whole and re-running
    # resumes correctly. Same .tmp+rename contract as SceneZipWriter and the
    # other downloaders in this directory.
    part = Path(str(filename) + ".part")
    try:
        ok = urlretrieve_multi_trials(url, part)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    if ok:
        os.replace(part, filename)
        return True
    part.unlink(missing_ok=True)
    return False


def check_download_file(cfg, url_template, remote_path, local_path, dry_run):
    """
    check if file exists, else download
    remote_path: relative to root
    local_path: full path
    dry_run: only check if file exists, don't download
    """
    # Updates: fix Windows users that use backslash for remote_path
    remote_path = str(remote_path).replace("\\", "/")
    url = url_template.replace("TOKEN", cfg.token).replace("FILEPATH", remote_path)

    if dry_run:
        status = check_remote_file_exists(url)
        if status:
            print("Remote file exists:", url)
        else:
            print("Remote file missing:", url)
        return status

    if local_path.is_file():
        if cfg.verbose:
            print("File exists, skipping download: ", local_path)
        return True
    else:
        return download_file(url, local_path, verbose=cfg.verbose, make_parent=True)


def main(args):
    print_banner(args.config_file)
    cfg = load_yaml_munch(args.config_file)
    # Ask the user to provide the token if not provided
    if cfg.get("token", "<YOUR_TOKEN_HERE>") == "<YOUR_TOKEN_HERE>":
        cfg.token = input("Please enter your download token: ").strip()
        if cfg.token == "":
            print("No token provided, exiting. Please apply for a token from ScanNet++ official website.")
            return 1
    else:
        # NOT the token itself: this runs under sbatch, whose stdout lands in a
        # world-readable logs/*.out that outlives the job. Print enough to tell
        # which token is in play and nothing that can be replayed with.
        tok = str(cfg.token)
        shown = f"{tok[:4]}...{tok[-4:]}" if len(tok) > 12 else "(too short to mask)"
        print(f"Using token from config file: {shown} ({len(tok)} chars)")

    if cfg.get("data_root", "<DOWNLOAD_LOCATION_HERE>") == "<DOWNLOAD_LOCATION_HERE>":
        cfg.data_root = input("Please enter your download location: (default: ./scannetpp_data)").strip()
        if cfg.data_root == "":
            cfg.data_root = "./scannetpp_data"
    print(f"Downloading to: {cfg.data_root}")

    # Notify user the size of the whole dataset
    print(
        "WARNING: Downloading the full ScanNet++ dataset with default splits and assets "
        "will require approximately 1.5TB of disk space.\n"
        "If you have already customized splits or assets in your config file, the required space may differ.\n"
        "Do you want to proceed with the download? (y/n)\n > ", end=""
    )
    ans = input().strip().lower()
    if ans != 'y':
        print("Exiting.")
        return 1

    if cfg.dry_run:
        print("Dry run: check if remote files exist, no files will be downloaded")

    missing = []

    data_root = Path(cfg.data_root)

    # create data root directory
    data_root.mkdir(parents=True, exist_ok=True)

    # download meta files
    for path in cfg.meta_files:
        if not check_download_file(cfg, cfg.root_url, path, data_root / path, cfg.dry_run):
            missing.append(str(data_root / path))

    if cfg.metadata_only:
        print("Downloaded metadata, done.")
        return 1 if missing else 0

    # read all the split files
    split_lists = {}
    for split in cfg.splits:
        split_path = data_root / "splits" / f"{split}.txt"
        split_lists[split] = read_txt_list(split_path)

    # get the list of scenes to be downloaded
    if cfg.get("download_scenes"):
        scene_ids = cfg.download_scenes
    elif cfg.get("download_splits"):
        scene_ids = []
        for split in cfg.download_splits:
            split_path = Path(cfg.data_root) / "splits" / f"{split}.txt"
            scene_ids += read_txt_list(split_path)
    else:
        # Neither key set, or download_scenes is an empty list (falsy). Without
        # this the next use of scene_ids raises a bare NameError.
        raise SystemExit(
            "config sets neither a non-empty `download_scenes` nor "
            "`download_splits`; there is nothing to download."
        )

    # we know the scene ids, check for 3rd party datasets
    if cfg.get("scannetpp_gs_dir"):
        download_scannetpp_gs(cfg, scene_ids)
        print(f"Downloaded ScanNet++GS data to {cfg.scannetpp_gs_dir}, done.")
        return 0

    # get the list of assets to download for these scenes
    if cfg.get("download_assets"):
        download_assets = cfg.download_assets
    elif cfg.get("download_options"):
        download_assets = []
        for option in cfg.download_options:
            option_assets = cfg.option_assets[option]
            for asset in option_assets:
                if asset not in download_assets:
                    download_assets.append(asset)
    else:
        download_assets = cfg.default_assets

    # Assets whose archive is kept instead of being extracted. A list, not a
    # flag, because the two kinds of zipped asset want opposite treatment:
    # a DIRECTORY asset (dslr/resized_images.zip) holds hundreds of frames, so
    # keeping it collapses them into one inode, while a single-FILE asset
    # (scans/mesh_aligned_0.05.zip, iphone/rgb_mask.zip) is only zip-wrapped
    # for transport -- keeping it saves no inodes and would force every reader
    # downstream to go through zipio for no gain. `true` keeps all of them,
    # `false`/absent restores the stock extract-and-delete behaviour.
    keep_zipped = cfg.get("keep_zipped") or []
    if isinstance(keep_zipped, bool):
        keep_zipped = list(cfg.zipped_assets) if keep_zipped else []
    unknown = set(keep_zipped) - set(cfg.zipped_assets)
    assert not unknown, f"keep_zipped names assets the server does not zip: {sorted(unknown)}"

    print("Downloading assets:", download_assets)
    print("Keeping zipped:", keep_zipped)
    print("Scenes selected: ", len(scene_ids))

    download_has_error = False
    for scene_id in tqdm(scene_ids, desc="scenes"):
        # download from here
        # path relative to root url
        src_scene = ScannetppScene_Release(scene_id, data_root="data")
        # to here
        tgt_scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / "data")

        # get the split for this scene
        #
        # The loop variable is deliberately NOT the result variable. Written as
        # `for split in cfg.splits: ... break`, a scene present in no split
        # leaves `split` bound to the LAST split, so the assert below can never
        # fire and the scene silently inherits that split's exclude_assets --
        # e.g. a scene falling through to nvs_test_iphone quietly skips every
        # dslr and scan asset, and the gap only surfaces during preprocessing.
        split = None
        for candidate in cfg.splits:
            if scene_id in split_lists[candidate]:
                split = candidate
                break
        assert split is not None, (
            f"Scene {scene_id} is not in any split listed in `splits`. It was "
            f"likely removed from the release; drop it from download_scenes."
        )

        for asset in tqdm(download_assets, desc="assets", leave=False):
            # some assets not present in test splits
            if asset in cfg.exclude_assets.get(split, []):
                continue

            # check if asset is zipped, download the zip and unzip it to the target path
            if asset in cfg.zipped_assets:
                tgt_path = getattr(tgt_scene, asset)
                tgt_download_path = tgt_path.with_suffix(".zip")

                # keep_zipped: the archive IS the deliverable, not a transport
                # wrapper. The preprocessing reads members straight out of it
                # via dust3r.utils.zipio, so extracting would trade ~4 inodes
                # per scene for the hundreds of thousands of loose files
                # inside -- 229 scenes of DSLR frames alone is well past the
                # per-user inode quota. The resume check therefore looks at
                # the archive rather than at the extracted directory.
                if asset in keep_zipped:
                    if zip_is_complete(tgt_download_path):
                        if cfg.verbose:
                            print("Archive exists, skipping download: ", tgt_download_path)
                        continue
                    if tgt_download_path.is_file() and not cfg.dry_run:
                        # Present but unreadable -- a pre-.part-rename download,
                        # or a bad transfer. check_download_file() skips
                        # anything that merely exists, so the corrupt archive
                        # would otherwise survive every re-run.
                        print("Removing incomplete archive:", tgt_download_path)
                        tgt_download_path.unlink()
                elif tgt_path.is_file() or tgt_path.is_dir():
                    if cfg.verbose:
                        print("File exists, skipping download: ", tgt_path)
                    continue

                src_download_path = getattr(src_scene, asset).with_suffix(".zip")

                if not check_download_file(cfg, cfg.root_url, src_download_path, tgt_download_path, cfg.dry_run):
                    missing.append(str(tgt_download_path))
                    # Abort the downloading process
                    download_has_error = True
                    break

                if not cfg.dry_run and asset not in keep_zipped:
                    # unzip it
                    if cfg.verbose:
                        print("Unzipping:", tgt_download_path)
                    with zipfile.ZipFile(tgt_download_path, "r") as zip_ref:
                        zip_ref.extractall(tgt_download_path.parent)
                    # remove the zip file
                    if cfg.verbose:
                        print("Delete zip file:", tgt_download_path)
                    tgt_download_path.unlink()
            else:
                #  download single file
                src_path = getattr(src_scene, asset)
                tgt_path = getattr(tgt_scene, asset)
                if not check_download_file(cfg, cfg.root_url, src_path, tgt_path, cfg.dry_run):
                    missing.append(str(tgt_path))
                    download_has_error = True
                    break

        if download_has_error:
            print(f"\nError downloading scene {scene_id}, aborting.")
            break

    if missing:
        # Exit non-zero: the caller (download_scannetpp.sh, or a bare
        # `python download_scannetpp.py cfg.yml`) has no other way to tell a
        # complete run from one that aborted on the first scene. Returning
        # None here made a 2-of-228 run report success.
        print(f"{len(missing)} files missing:", missing, file=sys.stderr)
        return 1
    print("Download successful!")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    sys.exit(main(args))

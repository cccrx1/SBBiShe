import argparse
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import DEFAULT_DATA_ROOT, prepare_gtsrb_testset, to_path
from prepare_cub200_dataset import prepare_cub200_dataset


GTSRB_URLS = {
    "train": "https://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip",
    "test_images": "https://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip",
    "test_gt": "https://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip",
}
CUB_DEFAULT_URLS = (
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
    "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz",
)

DEFAULT_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] already downloaded: {dst}")
        return
    ensure_dir(dst.parent)
    print(f"[download] {url} -> {dst}")
    temp_dst = dst.with_name(f"{dst.name}.part")
    if temp_dst.exists():
        temp_dst.unlink()

    request = urllib.request.Request(url, headers=DEFAULT_DOWNLOAD_HEADERS)
    try:
        with urllib.request.urlopen(request) as response, temp_dst.open("wb") as f:
            shutil.copyfileobj(response, f)
        temp_dst.replace(dst)
    except Exception:
        if temp_dst.exists():
            temp_dst.unlink()
        raise


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    ensure_dir(extract_to)
    print(f"[extract] {archive_path} -> {extract_to}")
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
        return

    suffixes = [s.lower() for s in archive_path.suffixes]
    if ".tgz" in suffixes or suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix.lower() == ".tar":
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_to)
        return

    raise ValueError(f"Unsupported archive format: {archive_path}")


def is_valid_tar_archive(archive_path: Path) -> bool:
    try:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.getmembers()
        return True
    except tarfile.TarError:
        return False


def copy_tree_if_needed(src: Path, dst: Path) -> int:
    if not src.exists():
        raise FileNotFoundError(f"Source path not found: {src}")
    ensure_dir(dst)

    copied = 0
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            ensure_dir(target)
            for item in child.iterdir():
                item_target = target / item.name
                if item_target.exists():
                    continue
                shutil.copy2(item, item_target)
                copied += 1
        else:
            if target.exists():
                continue
            shutil.copy2(child, target)
            copied += 1
    return copied


def setup_gtsrb(data_root: Path, cache_root: Path) -> None:
    gtsrb_train = data_root / "GTSRB" / "train"
    gtsrb_test = data_root / "GTSRB" / "testset"

    if gtsrb_train.exists() and any(gtsrb_train.iterdir()):
        print(f"[skip] GTSRB train already prepared: {gtsrb_train}")
    else:
        archives_root = cache_root / "gtsrb"
        extract_root = archives_root / "extracted"
        ensure_dir(archives_root)
        ensure_dir(extract_root)

        train_zip = archives_root / "GTSRB_Final_Training_Images.zip"
        test_zip = archives_root / "GTSRB_Final_Test_Images.zip"
        gt_zip = archives_root / "GTSRB_Final_Test_GT.zip"

        download_file(GTSRB_URLS["train"], train_zip)
        download_file(GTSRB_URLS["test_images"], test_zip)
        download_file(GTSRB_URLS["test_gt"], gt_zip)

        extract_archive(train_zip, extract_root)
        extract_archive(test_zip, extract_root)
        extract_archive(gt_zip, extract_root)

        source_train = extract_root / "GTSRB" / "Final_Training" / "Images"
        source_test = extract_root / "GTSRB" / "Final_Test" / "Images"

        copied_train = copy_tree_if_needed(source_train, gtsrb_train)
        copied_test = copy_tree_if_needed(source_test, gtsrb_test)
        print(f"[done] GTSRB copied train files: {copied_train}, test files: {copied_test}")

    prepared = prepare_gtsrb_testset(data_root)
    print(f"[done] GTSRB testset class folders prepared: {prepared}")


def resolve_cub_raw_root(source: str, cache_root: Path) -> Path:
    source_path = Path(source)
    if source_path.exists():
        if source_path.is_dir():
            return source_path

        archives_root = cache_root / "cub200"
        extract_root = archives_root / "extracted"
        ensure_dir(extract_root)
        extract_archive(source_path, extract_root)
        candidate = extract_root / "CUB_200_2011"
        if candidate.exists():
            return candidate
        return extract_root

    archives_root = cache_root / "cub200"
    ensure_dir(archives_root)
    archive_path = archives_root / "CUB_200_2011.tgz"

    if archive_path.exists() and not is_valid_tar_archive(archive_path):
        print(f"[warn] removing invalid cached archive: {archive_path}")
        archive_path.unlink()

    download_sources = [source]
    if source == CUB_DEFAULT_URLS[0]:
        download_sources.extend(url for url in CUB_DEFAULT_URLS[1:] if url not in download_sources)

    last_error = None
    for download_source in download_sources:
        try:
            download_file(download_source, archive_path)
            break
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            last_error = exc
            print(f"[warn] download failed, will try next source if available: {download_source} ({exc})")
    else:
        raise RuntimeError(f"Failed to download CUB-200-2011 from all sources: {download_sources}") from last_error

    extract_root = archives_root / "extracted"
    ensure_dir(extract_root)
    extract_archive(archive_path, extract_root)
    candidate = extract_root / "CUB_200_2011"
    if candidate.exists():
        return candidate
    return extract_root


def setup_cub200(data_root: Path, cache_root: Path, source: str) -> None:
    output_root = data_root / "CUB200"
    train_root = output_root / "train"
    test_root = output_root / "test"

    if train_root.exists() and any(train_root.iterdir()) and test_root.exists() and any(test_root.iterdir()):
        print(f"[skip] CUB200 already prepared: {output_root}")
        return

    raw_root = resolve_cub_raw_root(source, cache_root)
    copied = prepare_cub200_dataset(raw_root=raw_root, output_root=output_root, copy_mode="copy")
    print(f"[done] CUB200 prepared at {output_root}, copied files: {copied}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-download and prepare datasets into project-supported layout.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Dataset root, default: datasets")
    parser.add_argument("--dataset", choices=("gtsrb", "cub200", "all"), default="all")
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Download/extract cache directory. Defaults to <data-root>/_downloads",
    )
    parser.add_argument(
        "--cub-source",
        default=CUB_DEFAULT_URLS[0],
        help="CUB source: URL, local archive path, or extracted CUB_200_2011 directory",
    )
    args = parser.parse_args()

    data_root = to_path(args.data_root)
    cache_root = to_path(args.cache_root) if args.cache_root else (data_root / "_downloads")
    ensure_dir(data_root)
    ensure_dir(cache_root)

    if args.dataset in {"gtsrb", "all"}:
        setup_gtsrb(data_root=data_root, cache_root=cache_root)

    if args.dataset in {"cub200", "all"}:
        setup_cub200(data_root=data_root, cache_root=cache_root, source=args.cub_source)

    print("[done] bootstrap completed")


if __name__ == "__main__":
    main()

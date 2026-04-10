import argparse
import shutil
from pathlib import Path


def to_path(path_like):
    return Path(path_like).resolve()


def parse_mapping(file_path, split_char=" "):
    mapping = {}
    with file_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            key, value = raw.split(split_char, 1)
            mapping[int(key)] = value.strip()
    return mapping


def ensure_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required CUB metadata file not found: {path}")


def prepare_cub200_dataset(raw_root, output_root, copy_mode="copy"):
    raw_root = to_path(raw_root)
    output_root = to_path(output_root)

    images_root = raw_root / "images"
    images_txt = raw_root / "images.txt"
    split_txt = raw_root / "train_test_split.txt"
    classes_txt = raw_root / "classes.txt"

    for needed in (images_root, images_txt, split_txt, classes_txt):
        ensure_file(needed)

    id_to_relpath = parse_mapping(images_txt)
    id_to_split = parse_mapping(split_txt)
    id_to_class = parse_mapping(classes_txt)

    train_root = output_root / "train"
    test_root = output_root / "test"
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image_id, rel_path in id_to_relpath.items():
        if image_id not in id_to_split:
            raise RuntimeError(f"Missing split flag for image id {image_id}")

        src = images_root / rel_path
        if not src.exists():
            raise FileNotFoundError(f"CUB image not found: {src}")

        class_name = rel_path.split("/", 1)[0]
        split_flag = int(id_to_split[image_id])
        split_root = train_root if split_flag == 1 else test_root
        dst_dir = split_root / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name

        if dst.exists():
            continue

        if copy_mode != "copy":
            raise ValueError(f"Unsupported copy mode: {copy_mode}")
        shutil.copy2(src, dst)
        copied += 1

    return copied


def main():
    parser = argparse.ArgumentParser(description="Prepare CUB-200-2011 into DatasetFolder train/test layout.")
    parser.add_argument("--raw-root", required=True, help="Path to raw CUB_200_2011 root folder.")
    parser.add_argument("--output-root", default="datasets/CUB200", help="Output folder for processed CUB200.")
    parser.add_argument("--copy-mode", choices=("copy",), default="copy")
    args = parser.parse_args()

    copied = prepare_cub200_dataset(args.raw_root, args.output_root, copy_mode=args.copy_mode)
    print(f"Prepared CUB200 dataset. Copied files: {copied}")


if __name__ == "__main__":
    main()

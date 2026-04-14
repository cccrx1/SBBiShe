import argparse
import csv
import shutil
import os
import sys
from pathlib import Path

def _sanitize_thread_env():
    omp_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_threads is not None:
        try:
            if int(omp_threads) <= 0:
                raise ValueError
        except ValueError:
            os.environ["OMP_NUM_THREADS"] = "1"

    mkl_threads = os.environ.get("MKL_NUM_THREADS")
    if mkl_threads is not None:
        try:
            if int(mkl_threads) <= 0:
                raise ValueError
        except ValueError:
            os.environ["MKL_NUM_THREADS"] = "1"


_sanitize_thread_env()

import cv2
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
from torchvision.datasets import DatasetFolder


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import core


GLOBAL_SEED = 666
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "experiments"
DEFAULT_REFLECTION_DIR = DEFAULT_DATA_ROOT / "refool_reflections"
IMAGE_EXTENSIONS = ("png", "ppm", "jpg", "jpeg", "bmp")

DATASET_SPECS = {
    "gtsrb": {
        "display_name": "GTSRB",
        "num_classes": 43,
        "image_size": 32,
        "train_subdir": "GTSRB/train",
        "test_subdir": "GTSRB/testset",
        "task_type": "classification",
    },
    "cub200": {
        "display_name": "CUB200",
        "num_classes": 200,
        "image_size": 128,
        "train_subdir": "CUB200/train",
        "test_subdir": "CUB200/test",
        "task_type": "classification",
    },
}

ATTACK_NAMES = ("badnets", "blended", "wanet", "refool")
ATTACK_CANONICAL = {
    "badnets": "BadNets",
    "blended": "Blended",
    "wanet": "WaNet",
    "refool": "Refool",
}


def set_global_seed(seed=GLOBAL_SEED, deterministic=True):
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def to_path(path_like):
    return Path(path_like).resolve()


def normalize_pretrain_spec(spec):
    if spec is None:
        return None
    text = str(spec).strip()
    if text.lower().startswith("torchvision://"):
        return text
    return str(to_path(text))


def get_dataset_spec(dataset_name):
    normalized = str(dataset_name).strip().lower()
    if normalized not in DATASET_SPECS:
        supported = ", ".join(sorted(DATASET_SPECS.keys()))
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {supported}")
    return dict(DATASET_SPECS[normalized])


def add_dataset_args(parser):
    parser.add_argument(
        "--dataset",
        default="gtsrb",
        choices=tuple(DATASET_SPECS.keys()),
        help="Dataset name.",
    )
    return parser


def dataset_experiment_prefix(dataset_name):
    return str(dataset_name).strip().lower()


def parse_basic_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory that contains datasets.")
    parser.add_argument("--gpu-id", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--device", choices=("GPU", "CPU"), default="GPU")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--test-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--pretrain",
        default=None,
        help="Optional warm-start source: local checkpoint path or torchvision URI (e.g., torchvision://resnet18).",
    )
    return parser


def add_attack_training_args(parser):
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--schedule", action="append", type=int, default=None, help="Repeatable LR milestone, e.g. --schedule 20 --schedule 40")
    parser.add_argument("--disable-schedule", action="store_true", help="Disable LR milestone schedule and use an empty list.")
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    return parser


def add_refine_training_args(parser):
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--schedule", action="append", type=int, default=None, help="Repeatable LR milestone, e.g. --schedule 20 --schedule 40")
    parser.add_argument("--disable-schedule", action="store_true", help="Disable LR milestone schedule and use an empty list.")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--betas", nargs=2, type=float, default=None, metavar=("BETA1", "BETA2"))
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable mixed-precision training for REFINE on GPU.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed-precision training for REFINE.")
    parser.set_defaults(amp=True)
    return parser


def add_common_attack_args(parser, include_reflection=False):
    parser.add_argument("--y-target", type=int, default=None)
    parser.add_argument("--poisoned-rate", type=float, default=None)
    parser.add_argument("--trigger-size", type=int, default=None)
    parser.add_argument("--blended-alpha", type=float, default=None)
    parser.add_argument("--wanet-grid-k", type=int, default=None)
    parser.add_argument("--wanet-noise", dest="wanet_noise", action="store_true")
    parser.add_argument("--no-wanet-noise", dest="wanet_noise", action="store_false")
    parser.set_defaults(wanet_noise=None)
    if include_reflection:
        parser.add_argument("--reflection-dir", default=str(DEFAULT_REFLECTION_DIR))
        parser.add_argument("--reflection-limit", type=int, default=None)
    return parser


def build_resnet18(num_classes=None):
    if num_classes is None:
        num_classes = get_dataset_spec("gtsrb")["num_classes"]
    return core.models.ResNet(18, num_classes)


def load_image_bgr(path):
    return cv2.imread(str(path))


def make_gtsrb_transforms(attack_name=None, train=True, image_size=None):
    image_size = image_size or get_dataset_spec("gtsrb")["image_size"]
    if attack_name == "refool":
        items = [transforms.ToPILImage(), transforms.Resize((image_size, image_size))]
        if train:
            items.append(transforms.RandomHorizontalFlip())
        items.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        return transforms.Compose(items)

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def make_cub200_transforms(attack_name=None, train=True, image_size=None):
    image_size = image_size or get_dataset_spec("cub200")["image_size"]
    resize_size = int(image_size * 1.15)
    items = [transforms.ToPILImage(), transforms.Resize((resize_size, resize_size))]
    if train:
        items.append(transforms.RandomCrop((image_size, image_size)))
        items.append(transforms.RandomHorizontalFlip())
    else:
        items.append(transforms.CenterCrop((image_size, image_size)))

    items.append(transforms.ToTensor())
    items.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(items)


def make_dataset_transforms(dataset_name, attack_name=None, train=True, image_size=None):
    normalized = str(dataset_name).strip().lower()
    spec = get_dataset_spec(normalized)
    size = image_size if image_size is not None else spec["image_size"]
    if normalized == "gtsrb":
        return make_gtsrb_transforms(attack_name=attack_name, train=train, image_size=size)
    if normalized == "cub200":
        return make_cub200_transforms(attack_name=attack_name, train=train, image_size=size)
    raise ValueError(f"Unsupported dataset transform pipeline: {dataset_name}")


def ensure_datasetfolder_root(root, display_name, split_name):
    if not root.exists():
        raise FileNotFoundError(f"{display_name} {split_name} directory not found: {root}")
    if not any(root.iterdir()):
        raise FileNotFoundError(f"{display_name} {split_name} directory is empty: {root}")


def build_datasetfolder_pair(train_root, test_root, dataset_name, attack_name=None):
    trainset = DatasetFolder(
        root=str(train_root),
        loader=load_image_bgr,
        extensions=IMAGE_EXTENSIONS,
        transform=make_dataset_transforms(dataset_name, attack_name=attack_name, train=True),
        target_transform=None,
        is_valid_file=None,
    )
    testset = DatasetFolder(
        root=str(test_root),
        loader=load_image_bgr,
        extensions=IMAGE_EXTENSIONS,
        transform=make_dataset_transforms(dataset_name, attack_name=attack_name, train=False),
        target_transform=None,
        is_valid_file=None,
    )
    return trainset, testset


def load_gtsrb_datasets(data_root, attack_name=None):
    spec = get_dataset_spec("gtsrb")
    data_root = to_path(data_root)
    train_root = data_root / spec["train_subdir"]
    test_root = data_root / spec["test_subdir"]
    ensure_datasetfolder_root(train_root, spec["display_name"], "train")
    ensure_datasetfolder_root(test_root, spec["display_name"], "test")

    test_has_class_dirs = any(path.is_dir() for path in test_root.iterdir())
    test_csv = test_root / "GT-final_test.csv"
    if not test_has_class_dirs and test_csv.exists():
        raise FileNotFoundError(
            "GTSRB testset is still in raw flat layout. "
            f"Run `python scripts/prepare_gtsrb_testset.py --data-root {data_root}` first."
        )

    return build_datasetfolder_pair(train_root, test_root, "gtsrb", attack_name=attack_name)


def load_cub200_datasets(data_root, attack_name=None):
    spec = get_dataset_spec("cub200")
    data_root = to_path(data_root)
    train_root = data_root / spec["train_subdir"]
    test_root = data_root / spec["test_subdir"]
    ensure_datasetfolder_root(train_root, spec["display_name"], "train")
    ensure_datasetfolder_root(test_root, spec["display_name"], "test")
    return build_datasetfolder_pair(train_root, test_root, "cub200", attack_name=attack_name)


def load_datasets(dataset_name, data_root, attack_name=None, **kwargs):
    normalized = str(dataset_name).strip().lower()
    if normalized == "gtsrb":
        return load_gtsrb_datasets(data_root, attack_name=attack_name)
    if normalized == "cub200":
        return load_cub200_datasets(data_root, attack_name=attack_name)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def prepare_gtsrb_testset(data_root, copy_mode="copy"):
    data_root = to_path(data_root)
    test_root = data_root / "GTSRB" / "testset"
    csv_path = test_root / "GT-final_test.csv"
    if not test_root.exists():
        raise FileNotFoundError(f"GTSRB test directory not found: {test_root}")
    if not csv_path.exists():
        raise FileNotFoundError(f"GTSRB test CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    prepared = 0
    for row in rows:
        filename = row["Filename"]
        class_id = int(row["ClassId"])
        src = test_root / filename
        class_dir = test_root / f"{class_id:05d}"
        dst = class_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Expected GTSRB test image not found: {src}")
        class_dir.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        if copy_mode == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"Unsupported copy mode: {copy_mode}")
        prepared += 1

    return prepared


def build_badnets_pattern(image_size=None, trigger_size=3, alpha=1.0):
    size = image_size or get_dataset_spec("gtsrb")["image_size"]
    pattern = torch.zeros((size, size), dtype=torch.uint8)
    pattern[-trigger_size:, -trigger_size:] = 255
    weight = torch.zeros((size, size), dtype=torch.float32)
    weight[-trigger_size:, -trigger_size:] = alpha
    return pattern, weight


def poisoned_transform_indices(dataset_name, attack_name):
    dataset_name = str(dataset_name).strip().lower()
    attack_name = str(attack_name).strip().lower()

    if attack_name in {"badnets", "blended"}:
        if dataset_name == "cub200":
            return 4, 3
        return 2, 2

    if attack_name == "wanet":
        if dataset_name == "cub200":
            return 4, 3
        return 0, 0

    if attack_name == "refool":
        return 0, 0

    raise ValueError(f"Unsupported poisoned transform index for attack: {attack_name}")


def gen_wanet_grid(height, k=4):
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = nn.functional.interpolate(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)
    array1d = torch.linspace(-1, 1, steps=height)
    x, y = torch.meshgrid(array1d, array1d, indexing="ij")
    identity_grid = torch.stack((y, x), 2)[None, ...]
    return identity_grid, noise_grid


def get_wanet_grid_paths(experiment_root, dataset_name="gtsrb", image_size=None, k=4):
    dataset_name = str(dataset_name).strip().lower()
    image_size = image_size or get_dataset_spec(dataset_name)["image_size"]
    base = to_path(experiment_root) / "attacks" / "WaNet" / "grids"
    return (
        base / f"{dataset_name}_identity_grid_s{image_size}_k{k}.pth",
        base / f"{dataset_name}_noise_grid_s{image_size}_k{k}.pth",
    )


def ensure_wanet_grids(experiment_root, dataset_name="gtsrb", image_size=None, k=4):
    spec = get_dataset_spec(dataset_name)
    size = image_size or spec["image_size"]
    identity_path, noise_path = get_wanet_grid_paths(experiment_root, dataset_name=dataset_name, image_size=size, k=k)
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    if identity_path.exists() and noise_path.exists():
        return torch.load(identity_path), torch.load(noise_path), identity_path, noise_path

    identity_grid, noise_grid = gen_wanet_grid(height=size, k=k)
    torch.save(identity_grid, identity_path)
    torch.save(noise_grid, noise_path)
    return identity_grid, noise_grid, identity_path, noise_path


def load_reflection_images(reflection_dir, limit=200):
    reflection_dir = to_path(reflection_dir)
    if not reflection_dir.exists():
        raise FileNotFoundError(
            f"Refool reflection directory not found: {reflection_dir}. "
            "Create it and place reflection images there before training or evaluating Refool."
        )

    image_paths = sorted(
        [path for path in reflection_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )
    if not image_paths:
        raise FileNotFoundError(f"No reflection images found in: {reflection_dir}")

    images = []
    for path in image_paths[:limit]:
        image = cv2.imread(str(path))
        if image is not None:
            images.append(image)
    if not images:
        raise RuntimeError(f"Failed to load reflection images from: {reflection_dir}")
    return images


def default_attack_config(attack_name):
    return {
        "badnets": {"y_target": 1, "poisoned_rate": 0.05},
        "blended": {"y_target": 1, "poisoned_rate": 0.05, "trigger_size": 3, "blended_alpha": 0.2},
        "wanet": {"y_target": 0, "poisoned_rate": 0.1, "noise": True, "grid_k": 4},
        "refool": {"y_target": 1, "poisoned_rate": 0.05, "reflection_limit": 200},
    }[attack_name]


def attack_config(attack_name, args=None):
    config = dict(default_attack_config(attack_name))
    if args is None:
        return config

    if getattr(args, "y_target", None) is not None:
        config["y_target"] = args.y_target
    if getattr(args, "poisoned_rate", None) is not None:
        config["poisoned_rate"] = args.poisoned_rate
    if getattr(args, "trigger_size", None) is not None:
        config["trigger_size"] = args.trigger_size
    if getattr(args, "blended_alpha", None) is not None:
        config["blended_alpha"] = args.blended_alpha
    if getattr(args, "wanet_grid_k", None) is not None:
        config["grid_k"] = args.wanet_grid_k
    if getattr(args, "wanet_noise", None) is not None:
        config["noise"] = args.wanet_noise
    if getattr(args, "reflection_limit", None) is not None:
        config["reflection_limit"] = args.reflection_limit
    return config


def build_attack(
    attack_name,
    trainset,
    testset,
    experiment_root,
    dataset_name="gtsrb",
    reflection_dir=None,
    model=None,
    seed=GLOBAL_SEED,
    args=None,
):
    attack_name = attack_name.lower()
    spec = get_dataset_spec(dataset_name)
    config = attack_config(attack_name, args=args)
    model = model if model is not None else build_resnet18(spec["num_classes"])
    loss = nn.CrossEntropyLoss()

    if attack_name == "badnets":
        pattern, weight = build_badnets_pattern(
            image_size=spec["image_size"],
            trigger_size=config.get("trigger_size", 3),
            alpha=1.0,
        )
        poisoned_train_index, poisoned_test_index = poisoned_transform_indices(dataset_name, attack_name)
        return core.BadNets(
            train_dataset=trainset,
            test_dataset=testset,
            model=model,
            loss=loss,
            y_target=config["y_target"],
            poisoned_rate=config["poisoned_rate"],
            pattern=pattern,
            weight=weight,
            poisoned_transform_train_index=poisoned_train_index,
            poisoned_transform_test_index=poisoned_test_index,
            seed=seed,
            deterministic=True,
        )

    if attack_name == "blended":
        pattern, weight = build_badnets_pattern(
            image_size=spec["image_size"],
            trigger_size=config.get("trigger_size", 3),
            alpha=config.get("blended_alpha", 0.2),
        )
        poisoned_train_index, poisoned_test_index = poisoned_transform_indices(dataset_name, attack_name)
        return core.Blended(
            train_dataset=trainset,
            test_dataset=testset,
            model=model,
            loss=loss,
            y_target=config["y_target"],
            poisoned_rate=config["poisoned_rate"],
            pattern=pattern,
            weight=weight,
            poisoned_transform_train_index=poisoned_train_index,
            poisoned_transform_test_index=poisoned_test_index,
            seed=seed,
            deterministic=True,
        )

    if attack_name == "wanet":
        identity_grid, noise_grid, _, _ = ensure_wanet_grids(
            experiment_root,
            dataset_name=dataset_name,
            image_size=spec["image_size"],
            k=config.get("grid_k", 4),
        )
        poisoned_train_index, poisoned_test_index = poisoned_transform_indices(dataset_name, attack_name)
        return core.WaNet(
            train_dataset=trainset,
            test_dataset=testset,
            model=model,
            loss=loss,
            y_target=config["y_target"],
            poisoned_rate=config["poisoned_rate"],
            identity_grid=identity_grid,
            noise_grid=noise_grid,
            noise=config["noise"],
            poisoned_transform_train_index=poisoned_train_index,
            poisoned_transform_test_index=poisoned_test_index,
            seed=seed,
            deterministic=True,
        )

    if attack_name == "refool":
        reflection_images = load_reflection_images(
            reflection_dir or DEFAULT_REFLECTION_DIR,
            limit=config.get("reflection_limit", 200),
        )
        return core.Refool(
            train_dataset=trainset,
            test_dataset=testset,
            model=model,
            loss=loss,
            y_target=config["y_target"],
            poisoned_rate=config["poisoned_rate"],
            poisoned_transform_train_index=0,
            poisoned_transform_test_index=0,
            poisoned_target_transform_index=0,
            seed=seed,
            deterministic=True,
            reflection_candidates=reflection_images,
        )

    raise ValueError(f"Unsupported attack: {attack_name}")


def attack_output_root(experiment_root, attack_name):
    return to_path(experiment_root) / "attacks" / ATTACK_CANONICAL[attack_name]


def benign_output_root(experiment_root, dataset_name="gtsrb"):
    return to_path(experiment_root) / "benign"


def refine_output_root(experiment_root, attack_name, stage, dataset_name="gtsrb"):
    return to_path(experiment_root) / "refine" / ATTACK_CANONICAL[attack_name] / stage


def latest_timestamped_dir(root, prefix):
    root = to_path(root)
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def latest_checkpoint(root, prefix):
    root = to_path(root)
    if not root.exists():
        return None

    candidates = []
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        checkpoints = sorted(path.glob("ckpt_epoch_*.pth"))
        if checkpoints:
            candidates.append((path.stat().st_mtime, checkpoints[-1]))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def require_checkpoint(path_like, description):
    path = to_path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def resolve_schedule(default_schedule, args):
    if getattr(args, "disable_schedule", False):
        return []
    if getattr(args, "schedule", None) is not None:
        return list(args.schedule)
    return list(default_schedule)


def poisoned_rate_tag(poisoned_rate):
    text = f"{float(poisoned_rate):.6f}".rstrip("0").rstrip(".")
    return f"pr{text.replace('.', 'p')}"


def default_attack_schedule(args, benign_training, attack_name, save_dir, experiment_name):
    attack_name = attack_name.lower()
    dataset_name = str(getattr(args, "dataset", "gtsrb")).strip().lower()
    config = attack_config(attack_name, args=args)
    experiment_name_with_rate = experiment_name
    if not benign_training:
        experiment_name_with_rate = f"{experiment_name}_{poisoned_rate_tag(config['poisoned_rate'])}"
    schedule = {
        "device": args.device,
        "CUDA_VISIBLE_DEVICES": args.gpu_id,
        "GPU_num": 1,
        "benign_training": benign_training,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "log_iteration_interval": args.log_interval,
        "test_epoch_interval": args.test_interval,
        "save_epoch_interval": args.save_interval,
        "save_dir": str(save_dir),
        "experiment_name": experiment_name_with_rate,
    }

    if not benign_training:
        schedule["y_target"] = config["y_target"]
        schedule["poisoned_rate"] = config["poisoned_rate"]

    if attack_name in {"badnets", "blended"}:
        defaults = {"lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4, "gamma": 0.1, "schedule": [20, 60], "epochs": 100}
    elif attack_name == "wanet":
        defaults = {"lr": 0.01 if dataset_name == "cub200" else 0.1, "momentum": 0.9, "weight_decay": 5e-4, "gamma": 0.1, "schedule": [50, 75], "epochs": 100}
    elif attack_name == "refool":
        defaults = {"lr": 0.01 if dataset_name == "cub200" else 0.1, "momentum": 0.9, "weight_decay": 5e-4, "gamma": 0.1, "schedule": [50, 75], "epochs": 100}
    else:
        raise ValueError(f"Unsupported attack schedule for: {attack_name}")

    schedule.update(defaults)
    if args.lr is not None:
        schedule["lr"] = args.lr
    if args.momentum is not None:
        schedule["momentum"] = args.momentum
    if args.weight_decay is not None:
        schedule["weight_decay"] = args.weight_decay
    if args.gamma is not None:
        schedule["gamma"] = args.gamma
    if args.epochs is not None:
        schedule["epochs"] = args.epochs
    if args.pretrain is not None:
        schedule["pretrain"] = normalize_pretrain_spec(args.pretrain)
    schedule["schedule"] = resolve_schedule(defaults["schedule"], args)
    return schedule


def default_refine_schedule(args, attack_name, save_dir, dataset_name="gtsrb"):
    attack_name = attack_name.lower()
    dataset_name = str(dataset_name).strip().lower()
    config = attack_config(attack_name, args=args)
    prefix = dataset_experiment_prefix(dataset_name)
    experiment_name = f"{prefix}_refine_{attack_name}_train_{poisoned_rate_tag(config['poisoned_rate'])}"
    default_lr = 0.001 if dataset_name == "cub200" else 0.01
    defaults = {
        "device": args.device,
        "CUDA_VISIBLE_DEVICES": args.gpu_id,
        "GPU_num": 1,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": default_lr,
        "betas": (0.9, 0.999),
        "eps": 1e-08,
        "weight_decay": 0,
        "amsgrad": False,
        "schedule": [100, 130],
        "gamma": 0.1,
        "epochs": 150,
        "log_iteration_interval": args.log_interval,
        "test_epoch_interval": args.test_interval,
        "save_epoch_interval": args.save_interval,
        "save_dir": str(save_dir),
        "experiment_name": experiment_name,
        "y_target": config["y_target"],
        "poisoned_rate": config["poisoned_rate"],
        "amp": True,
    }
    schedule = dict(defaults)
    if args.lr is not None:
        schedule["lr"] = args.lr
    if args.weight_decay is not None:
        schedule["weight_decay"] = args.weight_decay
    if args.gamma is not None:
        schedule["gamma"] = args.gamma
    if args.epochs is not None:
        schedule["epochs"] = args.epochs
    if args.betas is not None:
        schedule["betas"] = tuple(args.betas)
    if args.eps is not None:
        schedule["eps"] = args.eps
    if args.amsgrad:
        schedule["amsgrad"] = True
    if getattr(args, "amp", None) is not None:
        schedule["amp"] = bool(args.amp)
    if args.pretrain is not None:
        schedule["pretrain"] = normalize_pretrain_spec(args.pretrain)
    schedule["schedule"] = resolve_schedule(defaults["schedule"], args)
    return schedule


def load_model_checkpoint(model, checkpoint_path):
    checkpoint_path = require_checkpoint(checkpoint_path, "Model checkpoint")
    model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"), strict=False)
    return model


def build_refine_defense(model, num_classes, checkpoint_path=None, arr_path=None, seed=GLOBAL_SEED):
    lmd = 0.05 if num_classes >= 100 else 0.1
    supcon_temperature = 0.2 if num_classes >= 100 else 0.07
    enable_label_shuffle = num_classes < 100
    return core.REFINE(
        unet=core.models.UNetLittle(args=None, n_channels=3, n_classes=3, first_channels=64),
        model=model,
        pretrain=str(checkpoint_path) if checkpoint_path else None,
        arr_path=str(arr_path) if arr_path else None,
        num_classes=num_classes,
        lmd=lmd,
        supcon_temperature=supcon_temperature,
        enable_label_shuffle=enable_label_shuffle,
        seed=seed,
        deterministic=True,
    )


def infer_attack_checkpoint(experiment_root, attack_name, poisoned_rate=None, dataset_name="gtsrb"):
    prefix_dataset = dataset_experiment_prefix(dataset_name)
    base_prefix = f"{prefix_dataset}_{attack_name}"
    prefix = base_prefix
    if poisoned_rate is not None:
        prefix = f"{prefix}_{poisoned_rate_tag(poisoned_rate)}"
    path = latest_checkpoint(attack_output_root(experiment_root, attack_name), prefix)
    if path is None and poisoned_rate is not None:
        path = latest_checkpoint(attack_output_root(experiment_root, attack_name), base_prefix)
    if path is None:
        raise FileNotFoundError(
            f"No trained checkpoint found for {ATTACK_CANONICAL[attack_name]} under "
            f"{attack_output_root(experiment_root, attack_name)}"
        )
    return path


def infer_refine_artifacts(experiment_root, attack_name, poisoned_rate=None, dataset_name="gtsrb"):
    prefix_dataset = dataset_experiment_prefix(dataset_name)
    prefix = f"{prefix_dataset}_refine_{attack_name}_train"
    if poisoned_rate is not None:
        prefix = f"{prefix}_{poisoned_rate_tag(poisoned_rate)}"
    exp_dir = latest_timestamped_dir(refine_output_root(experiment_root, attack_name, "train"), prefix)
    if exp_dir is None:
        raise FileNotFoundError(
            f"No REFINE training run found for {ATTACK_CANONICAL[attack_name]} under "
            f"{refine_output_root(experiment_root, attack_name, 'train')}"
        )
    checkpoints = sorted(exp_dir.glob("ckpt_epoch_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No REFINE checkpoints found in: {exp_dir}")
    arr_path = exp_dir / "label_shuffle.pth"
    if not arr_path.exists():
        raise FileNotFoundError(f"REFINE label shuffle file not found: {arr_path}")
    return checkpoints[-1], arr_path, exp_dir


def manual_refine_eval(defense, dataset, device, batch_size, y_target=None, ignore_target=False, label_dataset=None):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )
    label_loader = None
    if label_dataset is not None:
        label_loader = torch.utils.data.DataLoader(
            label_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True,
        )
    defense.unet = defense.unet.to(device)
    defense.unet.eval()
    defense.model = defense.model.to(device)
    defense.model.eval()

    total = 0
    correct = 0
    if label_loader is None:
        iterator = ((imgs, labels, labels) for imgs, labels in loader)
    else:
        iterator = ((imgs, labels, clean_labels) for (imgs, labels), (_, clean_labels) in zip(loader, label_loader))

    for batch_img, batch_label, batch_clean_label in iterator:
        batch_img = batch_img.to(device)
        logits = defense.forward(batch_img).cpu()
        preds = logits.argmax(dim=1)
        mask = torch.ones_like(batch_clean_label, dtype=torch.bool)
        if ignore_target and y_target is not None:
            mask = batch_clean_label != y_target
        if mask.sum().item() == 0:
            continue
        filtered_preds = preds[mask]
        filtered_labels = batch_label[mask]
        correct += (filtered_preds == filtered_labels).sum().item()
        total += filtered_labels.size(0)
    return correct, total, (correct / total if total else 0.0)

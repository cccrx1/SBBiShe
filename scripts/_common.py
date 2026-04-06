import argparse
import os
import sys
from pathlib import Path

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


NUM_CLASSES = 43
IMAGE_SIZE = 32
GLOBAL_SEED = 666
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "experiments"
DEFAULT_REFLECTION_DIR = DEFAULT_DATA_ROOT / "refool_reflections"

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


def parse_basic_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory that contains GTSRB.")
    parser.add_argument("--gpu-id", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--device", choices=("GPU", "CPU"), default="GPU")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--test-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser


def build_resnet18():
    return core.models.ResNet(18, NUM_CLASSES)


def load_image_bgr(path):
    return cv2.imread(str(path))


def make_gtsrb_transforms(attack_name=None, train=True):
    if attack_name == "refool":
        items = [transforms.ToPILImage(), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
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
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )


def load_gtsrb_datasets(data_root, attack_name=None):
    data_root = to_path(data_root)
    train_root = data_root / "GTSRB" / "train"
    test_root = data_root / "GTSRB" / "testset"
    if not train_root.exists():
        raise FileNotFoundError(f"GTSRB train directory not found: {train_root}")
    if not test_root.exists():
        raise FileNotFoundError(f"GTSRB test directory not found: {test_root}")

    trainset = DatasetFolder(
        root=str(train_root),
        loader=load_image_bgr,
        extensions=("png",),
        transform=make_gtsrb_transforms(attack_name, train=True),
        target_transform=None,
        is_valid_file=None,
    )
    testset = DatasetFolder(
        root=str(test_root),
        loader=load_image_bgr,
        extensions=("png",),
        transform=make_gtsrb_transforms(attack_name, train=False),
        target_transform=None,
        is_valid_file=None,
    )
    return trainset, testset


def build_badnets_pattern(alpha=1.0):
    pattern = torch.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    weight[-3:, -3:] = alpha
    return pattern, weight


def gen_wanet_grid(height=IMAGE_SIZE, k=4):
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = nn.functional.interpolate(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)
    array1d = torch.linspace(-1, 1, steps=height)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...]
    return identity_grid, noise_grid


def get_wanet_grid_paths(experiment_root):
    base = to_path(experiment_root) / "attacks" / "WaNet" / "grids"
    return base / "gtsrb_identity_grid.pth", base / "gtsrb_noise_grid.pth"


def ensure_wanet_grids(experiment_root):
    identity_path, noise_path = get_wanet_grid_paths(experiment_root)
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    if identity_path.exists() and noise_path.exists():
        return torch.load(identity_path), torch.load(noise_path), identity_path, noise_path

    identity_grid, noise_grid = gen_wanet_grid()
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


def attack_config(attack_name):
    return {
        "badnets": {"y_target": 1, "poisoned_rate": 0.05},
        "blended": {"y_target": 1, "poisoned_rate": 0.05},
        "wanet": {"y_target": 0, "poisoned_rate": 0.1, "noise": True},
        "refool": {"y_target": 1, "poisoned_rate": 0.05},
    }[attack_name]


def build_attack(attack_name, trainset, testset, experiment_root, reflection_dir=None, model=None, seed=GLOBAL_SEED):
    attack_name = attack_name.lower()
    config = attack_config(attack_name)
    model = model if model is not None else build_resnet18()
    loss = nn.CrossEntropyLoss()

    if attack_name == "badnets":
        pattern, weight = build_badnets_pattern(alpha=1.0)
        return core.BadNets(
            train_dataset=trainset,
            test_dataset=testset,
            model=model,
            loss=loss,
            y_target=config["y_target"],
            poisoned_rate=config["poisoned_rate"],
            pattern=pattern,
            weight=weight,
            poisoned_transform_train_index=2,
            poisoned_transform_test_index=2,
            seed=seed,
            deterministic=True,
        )

    if attack_name == "blended":
        pattern, weight = build_badnets_pattern(alpha=0.2)
        return core.Blended(
            train_dataset=trainset,
            test_dataset=testset,
            model=model,
            loss=loss,
            y_target=config["y_target"],
            poisoned_rate=config["poisoned_rate"],
            pattern=pattern,
            weight=weight,
            poisoned_transform_train_index=2,
            poisoned_transform_test_index=2,
            seed=seed,
            deterministic=True,
        )

    if attack_name == "wanet":
        identity_grid, noise_grid, _, _ = ensure_wanet_grids(experiment_root)
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
            seed=seed,
            deterministic=True,
        )

    if attack_name == "refool":
        reflection_images = load_reflection_images(reflection_dir or DEFAULT_REFLECTION_DIR)
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


def benign_output_root(experiment_root):
    return to_path(experiment_root) / "benign"


def refine_output_root(experiment_root, attack_name, stage):
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
    exp_dir = latest_timestamped_dir(root, prefix)
    if exp_dir is None:
        return None
    checkpoints = sorted(exp_dir.glob("ckpt_epoch_*.pth"))
    if not checkpoints:
        return None
    return checkpoints[-1]


def require_checkpoint(path_like, description):
    path = to_path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def default_attack_schedule(args, benign_training, attack_name, save_dir, experiment_name):
    attack_name = attack_name.lower()
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
        "experiment_name": experiment_name,
    }

    if attack_name in {"badnets", "blended"}:
        schedule.update(
            {
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 5e-4,
                "gamma": 0.1,
                "schedule": [20],
                "epochs": 30,
            }
        )
    elif attack_name == "wanet":
        schedule.update(
            {
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 5e-4,
                "gamma": 0.1,
                "schedule": [150, 180],
                "epochs": 200,
            }
        )
    elif attack_name == "refool":
        schedule.update(
            {
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 5e-4,
                "gamma": 0.1,
                "schedule": [50, 75],
                "epochs": 100,
            }
        )
    else:
        raise ValueError(f"Unsupported attack schedule for: {attack_name}")
    return schedule


def default_refine_schedule(args, attack_name, save_dir):
    return {
        "device": args.device,
        "CUDA_VISIBLE_DEVICES": args.gpu_id,
        "GPU_num": 1,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": 0.01,
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
        "experiment_name": f"gtsrb_refine_{attack_name}_train",
    }


def load_model_checkpoint(model, checkpoint_path):
    checkpoint_path = require_checkpoint(checkpoint_path, "Model checkpoint")
    model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"), strict=False)
    return model


def build_refine_defense(model, checkpoint_path=None, arr_path=None, seed=GLOBAL_SEED):
    return core.REFINE(
        unet=core.models.UNetLittle(args=None, n_channels=3, n_classes=3, first_channels=64),
        model=model,
        pretrain=str(checkpoint_path) if checkpoint_path else None,
        arr_path=str(arr_path) if arr_path else None,
        num_classes=NUM_CLASSES,
        seed=seed,
        deterministic=True,
    )


def infer_attack_checkpoint(experiment_root, attack_name):
    path = latest_checkpoint(attack_output_root(experiment_root, attack_name), f"gtsrb_{attack_name}")
    if path is None:
        raise FileNotFoundError(
            f"No trained checkpoint found for {ATTACK_CANONICAL[attack_name]} under "
            f"{attack_output_root(experiment_root, attack_name)}"
        )
    return path


def infer_refine_artifacts(experiment_root, attack_name):
    exp_dir = latest_timestamped_dir(refine_output_root(experiment_root, attack_name, "train"), f"gtsrb_refine_{attack_name}_train")
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


def manual_refine_eval(defense, dataset, device, batch_size, y_target=None, ignore_target=False):
    loader = torch.utils.data.DataLoader(
        dataset,
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
    for batch_img, batch_label in loader:
        batch_img = batch_img.to(device)
        logits = defense.forward(batch_img).cpu()
        preds = logits.argmax(dim=1)
        mask = torch.ones_like(batch_label, dtype=torch.bool)
        if ignore_target and y_target is not None:
            mask = batch_label != y_target
        if mask.sum().item() == 0:
            continue
        filtered_preds = preds[mask]
        filtered_labels = batch_label[mask]
        correct += (filtered_preds == filtered_labels).sum().item()
        total += filtered_labels.size(0)
    return correct, total, (correct / total if total else 0.0)

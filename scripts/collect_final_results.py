import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


METRIC_RE = re.compile(r"^(?P<name>BA|ASR_NoTarget)=(?P<value>[0-9]*\.?[0-9]+)\s*\((?P<corr>\d+)\/(?P<tot>\d+)\)")
RATE_TAG_RE = re.compile(r"_pr(?P<tag>[0-9p]+)")


@dataclass
class FinalResult:
    dataset: str
    attack: str
    poisoned_rate: Optional[float]
    ba: Optional[float]
    ba_correct: Optional[int]
    ba_total: Optional[int]
    asr_no_target: Optional[float]
    asr_correct: Optional[int]
    asr_total: Optional[int]
    attack_checkpoint: str
    refine_checkpoint: str
    arr_path: str
    result_file: str


def to_path(path_like: str) -> Path:
    return Path(path_like).resolve()


def parse_rate_tag(text: str) -> Optional[float]:
    match = RATE_TAG_RE.search(text)
    if not match:
        return None
    raw = match.group("tag").replace("p", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def parse_results_file(file_path: Path) -> Dict[str, object]:
    kv: Dict[str, object] = {
        "attack": "",
        "poisoned_rate": None,
        "attack_checkpoint": "",
        "refine_checkpoint": "",
        "label_shuffle": "",
        "BA": None,
        "BA_correct": None,
        "BA_total": None,
        "ASR_NoTarget": None,
        "ASR_NoTarget_correct": None,
        "ASR_NoTarget_total": None,
    }

    for raw in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue

        metric_match = METRIC_RE.match(line)
        if metric_match:
            name = metric_match.group("name")
            kv[name] = float(metric_match.group("value"))
            kv[f"{name}_correct"] = int(metric_match.group("corr"))
            kv[f"{name}_total"] = int(metric_match.group("tot"))
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key == "poisoned_rate":
            try:
                kv[key] = float(value)
            except ValueError:
                kv[key] = None
        elif key in kv:
            kv[key] = value

    return kv


def infer_dataset_from_eval_dir(eval_dir_name: str) -> str:
    parts = eval_dir_name.split("_")
    return parts[0].lower() if parts else "unknown"


def collect_refine_eval_results(experiment_root: Path, latest_only: bool, dataset_filter: Optional[str]) -> List[FinalResult]:
    base = experiment_root / "refine"
    if not base.exists():
        return []

    rows: List[FinalResult] = []
    for results_file in base.glob("*/eval/*/results.txt"):
        eval_dir = results_file.parent
        eval_dir_name = eval_dir.name
        if latest_only and "latest" not in eval_dir_name:
            continue

        data = parse_results_file(results_file)
        dataset = infer_dataset_from_eval_dir(eval_dir_name)
        if dataset_filter and dataset != dataset_filter:
            continue

        poisoned_rate = data["poisoned_rate"]
        if poisoned_rate is None:
            poisoned_rate = parse_rate_tag(eval_dir_name)

        rows.append(
            FinalResult(
                dataset=dataset,
                attack=str(data["attack"]),
                poisoned_rate=poisoned_rate,
                ba=data["BA"],
                ba_correct=data["BA_correct"],
                ba_total=data["BA_total"],
                asr_no_target=data["ASR_NoTarget"],
                asr_correct=data["ASR_NoTarget_correct"],
                asr_total=data["ASR_NoTarget_total"],
                attack_checkpoint=str(data["attack_checkpoint"]),
                refine_checkpoint=str(data["refine_checkpoint"]),
                arr_path=str(data["label_shuffle"]),
                result_file=str(results_file),
            )
        )

    rows.sort(
        key=lambda row: (
            row.dataset,
            row.attack,
            row.poisoned_rate if row.poisoned_rate is not None else -1,
            row.result_file,
        )
    )
    return rows


def write_csv(rows: List[FinalResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "attack",
        "poisoned_rate",
        "ba",
        "ba_correct",
        "ba_total",
        "asr_no_target",
        "asr_correct",
        "asr_total",
        "attack_checkpoint",
        "refine_checkpoint",
        "arr_path",
        "result_file",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(rows: List[FinalResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def write_markdown(rows: List[FinalResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Dataset | Attack | Poisoned Rate | BA | ASR_NoTarget | BA Count | ASR Count |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        rate = "" if row.poisoned_rate is None else str(row.poisoned_rate)
        ba_count = "" if row.ba_correct is None or row.ba_total is None else f"{row.ba_correct}/{row.ba_total}"
        asr_count = "" if row.asr_correct is None or row.asr_total is None else f"{row.asr_correct}/{row.asr_total}"
        lines.append(
            f"| {row.dataset} | {row.attack} | {rate} | {format_float(row.ba)} | {format_float(row.asr_no_target)} | {ba_count} | {asr_count} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rate_key(rate: Optional[float]) -> float:
    return -1.0 if rate is None else float(rate)


def rate_label(rate: Optional[float]) -> str:
    if rate is None:
        return "NA"
    return f"{rate:.6f}".rstrip("0").rstrip(".")


def build_paper_wide_rows(rows: List[FinalResult]) -> tuple[list[str], list[Dict[str, object]]]:
    rates = sorted({row.poisoned_rate for row in rows}, key=rate_key)

    columns = ["dataset", "attack"]
    for rate in rates:
        tag = rate_label(rate)
        columns.append(f"BA@{tag}")
        columns.append(f"ASR_NoTarget@{tag}")

    latest_by_key: Dict[tuple[str, str, Optional[float]], FinalResult] = {}
    for row in rows:
        key = (row.dataset, row.attack, row.poisoned_rate)
        latest_by_key[key] = row

    group_keys = sorted({(row.dataset, row.attack) for row in rows})
    table_rows: list[Dict[str, object]] = []
    for dataset, attack in group_keys:
        item: Dict[str, object] = {"dataset": dataset, "attack": attack}
        for rate in rates:
            tag = rate_label(rate)
            result = latest_by_key.get((dataset, attack, rate))
            item[f"BA@{tag}"] = "" if result is None or result.ba is None else f"{result.ba:.6f}"
            item[f"ASR_NoTarget@{tag}"] = "" if result is None or result.asr_no_target is None else f"{result.asr_no_target:.6f}"
        table_rows.append(item)

    return columns, table_rows


def write_paper_markdown(rows: List[FinalResult], output_path: Path) -> None:
    columns, table_rows = build_paper_wide_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in table_rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_csv(rows: List[FinalResult], output_path: Path) -> None:
    columns, table_rows = build_paper_wide_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)


def detect_output_format(output_path: Path, explicit_format: Optional[str]) -> str:
    if explicit_format:
        return explicit_format
    suffix = output_path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in {".md", ".markdown"}:
        return "md"
    return "csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect REFINE final eval results into one summary file.")
    parser.add_argument("--experiment-root", default="experiments", help="Root directory of experiment outputs.")
    parser.add_argument("--dataset", default=None, help="Optional dataset filter, e.g. gtsrb or cub200.")
    parser.add_argument("--latest-only", action="store_true", help="Only include eval folders with 'latest' in their name.")
    parser.add_argument("--format", choices=("csv", "json", "md"), default=None, help="Output format. Defaults by file extension.")
    parser.add_argument(
        "--output",
        default="experiments/final_summary/final_results.csv",
        help="Output summary file path.",
    )
    parser.add_argument(
        "--paper-output",
        default=None,
        help="Optional paper-style wide table output path (Attack x Poisoned Rate).",
    )
    parser.add_argument(
        "--paper-format",
        choices=("md", "csv"),
        default="md",
        help="Paper output format when --paper-output is provided.",
    )
    args = parser.parse_args()

    experiment_root = to_path(args.experiment_root)
    output_path = to_path(args.output)

    rows = collect_refine_eval_results(
        experiment_root=experiment_root,
        latest_only=args.latest_only,
        dataset_filter=args.dataset.lower() if args.dataset else None,
    )

    output_format = detect_output_format(output_path, args.format)
    if output_format == "csv":
        write_csv(rows, output_path)
    elif output_format == "json":
        write_json(rows, output_path)
    else:
        write_markdown(rows, output_path)

    print(f"Collected {len(rows)} records -> {output_path}")

    if args.paper_output:
        paper_output_path = to_path(args.paper_output)
        if args.paper_format == "csv":
            write_paper_csv(rows, paper_output_path)
        else:
            write_paper_markdown(rows, paper_output_path)
        print(f"Paper wide table -> {paper_output_path}")


if __name__ == "__main__":
    main()

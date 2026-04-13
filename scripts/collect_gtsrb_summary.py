import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, List, Optional


TOP1_RE = re.compile(
    r"Top-1 correct / Total: (?P<correct>\d+)/(?P<total>\d+), Top-1 accuracy: (?P<acc>[0-9]*\.?[0-9]+)"
)
REFINE_METRIC_RE = re.compile(r"^(?P<name>BA|ASR_NoTarget)=(?P<value>[0-9]*\.?[0-9]+)\s*\((?P<corr>\d+)\/(?P<tot>\d+)\)")
RATE_TAG_RE = re.compile(r"_pr(?P<tag>[0-9p]+)")


@dataclass
class SummaryRow:
    kind: str
    dataset: str
    attack: str
    poisoned_rate: Optional[float]
    clean_metric: str
    clean_accuracy: Optional[float]
    clean_correct: Optional[int]
    clean_total: Optional[int]
    poisoned_metric: str
    poisoned_accuracy: Optional[float]
    poisoned_correct: Optional[int]
    poisoned_total: Optional[int]
    checkpoint: str
    attack_checkpoint: str
    refine_checkpoint: str
    arr_path: str
    source_file: str


@dataclass
class CompareRow:
    dataset: str
    attack: str
    poisoned_rate: Optional[float]
    clean_before: Optional[float]
    asr_before: Optional[float]
    ba_after: Optional[float]
    asr_after: Optional[float]
    delta_clean: Optional[float]
    delta_asr: Optional[float]
    attack_source: str
    refine_source: str


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


def infer_dataset(text: str) -> str:
    lowered = text.lower()
    if "cub200" in lowered:
        return "cub200"
    if "gtsrb" in lowered:
        return "gtsrb"
    return "unknown"


def format_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def load_lines(path: Path) -> Iterable[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def checkpoint_from_dir(root: Path) -> str:
    checkpoints = sorted(root.glob("ckpt_epoch_*.pth"))
    if not checkpoints:
        return ""
    return str(checkpoints[-1])


def parse_test_log(log_path: Path, kind: str) -> Dict[str, Optional[object]]:
    current_section: Optional[str] = None
    clean: Dict[str, Optional[object]] = {"accuracy": None, "correct": None, "total": None}
    poisoned: Dict[str, Optional[object]] = {"accuracy": None, "correct": None, "total": None}

    for raw in load_lines(log_path):
        line = raw.strip()
        if not line:
            continue

        lowered = line.lower()
        if "test result on benign test dataset" in lowered:
            current_section = "clean"
            continue
        if "test result on poisoned test dataset" in lowered:
            current_section = "poisoned"
            continue

        match = TOP1_RE.search(line)
        if not match or current_section not in {"clean", "poisoned"}:
            continue

        payload = {
            "accuracy": float(match.group("acc")),
            "correct": int(match.group("correct")),
            "total": int(match.group("total")),
        }
        if current_section == "clean":
            clean.update(payload)
        else:
            poisoned.update(payload)

    dataset = infer_dataset(log_path.parent.name)
    poison_metric = "PoisonedEval" if kind == "benign" else "ASR"
    return {
        "dataset": dataset,
        "attack": "benign" if kind == "benign" else log_path.parent.name,
        "poisoned_rate": parse_rate_tag(log_path.parent.name),
        "clean_metric": "Top-1",
        "clean_accuracy": clean["accuracy"],
        "clean_correct": clean["correct"],
        "clean_total": clean["total"],
        "poisoned_metric": poison_metric,
        "poisoned_accuracy": poisoned["accuracy"],
        "poisoned_correct": poisoned["correct"],
        "poisoned_total": poisoned["total"],
        "checkpoint": checkpoint_from_dir(log_path.parent),
        "attack_checkpoint": "",
        "refine_checkpoint": "",
        "arr_path": "",
        "source_file": str(log_path),
    }


def parse_refine_results(results_path: Path) -> Dict[str, Optional[object]]:
    data: Dict[str, Optional[object]] = {
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

    for raw in load_lines(results_path):
        line = raw.strip()
        if not line:
            continue

        metric_match = REFINE_METRIC_RE.match(line)
        if metric_match:
            name = metric_match.group("name")
            data[name] = float(metric_match.group("value"))
            data[f"{name}_correct"] = int(metric_match.group("corr"))
            data[f"{name}_total"] = int(metric_match.group("tot"))
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "poisoned_rate":
            try:
                data[key] = float(value)
            except ValueError:
                data[key] = None
        elif key in data:
            data[key] = value

    eval_dir_name = results_path.parent.name
    return {
        "dataset": infer_dataset(eval_dir_name),
        "attack": str(data["attack"]) or eval_dir_name,
        "poisoned_rate": data["poisoned_rate"] if data["poisoned_rate"] is not None else parse_rate_tag(eval_dir_name),
        "clean_metric": "BA",
        "clean_accuracy": data["BA"],
        "clean_correct": data["BA_correct"],
        "clean_total": data["BA_total"],
        "poisoned_metric": "ASR_NoTarget",
        "poisoned_accuracy": data["ASR_NoTarget"],
        "poisoned_correct": data["ASR_NoTarget_correct"],
        "poisoned_total": data["ASR_NoTarget_total"],
        "checkpoint": "",
        "attack_checkpoint": str(data["attack_checkpoint"]),
        "refine_checkpoint": str(data["refine_checkpoint"]),
        "arr_path": str(data["label_shuffle"]),
        "source_file": str(results_path),
    }


def collect_benign_rows(experiment_root: Path, dataset_filter: Optional[str]) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    benign_root = experiment_root / "benign"
    if not benign_root.exists():
        return rows

    for log_path in benign_root.glob("*/log.txt"):
        if "_eval_" in log_path.parent.name:
            continue
        parsed = parse_test_log(log_path, kind="benign")
        if dataset_filter and parsed["dataset"] != dataset_filter:
            continue
        if parsed["clean_accuracy"] is None:
            continue
        rows.append(
            SummaryRow(
                kind="benign",
                dataset=str(parsed["dataset"]),
                attack="benign",
                poisoned_rate=None,
                clean_metric=str(parsed["clean_metric"]),
                clean_accuracy=parsed["clean_accuracy"],
                clean_correct=parsed["clean_correct"],
                clean_total=parsed["clean_total"],
                poisoned_metric=str(parsed["poisoned_metric"]),
                poisoned_accuracy=parsed["poisoned_accuracy"],
                poisoned_correct=parsed["poisoned_correct"],
                poisoned_total=parsed["poisoned_total"],
                checkpoint=str(parsed["checkpoint"]),
                attack_checkpoint="",
                refine_checkpoint="",
                arr_path="",
                source_file=str(parsed["source_file"]),
            )
        )

    rows.sort(key=lambda row: (row.dataset, row.source_file))
    return rows


def collect_attack_rows(experiment_root: Path, dataset_filter: Optional[str]) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    attack_root = experiment_root / "attacks"
    if not attack_root.exists():
        return rows

    for attack_dir in attack_root.iterdir():
        if not attack_dir.is_dir():
            continue
        for log_path in attack_dir.glob("*/log.txt"):
            if "_eval_" in log_path.parent.name:
                continue
            parsed = parse_test_log(log_path, kind="attack")
            if dataset_filter and parsed["dataset"] != dataset_filter:
                continue
            if parsed["clean_accuracy"] is None or parsed["poisoned_accuracy"] is None:
                continue
            rows.append(
                SummaryRow(
                    kind="attack",
                    dataset=str(parsed["dataset"]),
                    attack=str(attack_dir.name),
                    poisoned_rate=parsed["poisoned_rate"],
                    clean_metric=str(parsed["clean_metric"]),
                    clean_accuracy=parsed["clean_accuracy"],
                    clean_correct=parsed["clean_correct"],
                    clean_total=parsed["clean_total"],
                    poisoned_metric=str(parsed["poisoned_metric"]),
                    poisoned_accuracy=parsed["poisoned_accuracy"],
                    poisoned_correct=parsed["poisoned_correct"],
                    poisoned_total=parsed["poisoned_total"],
                    checkpoint=str(parsed["checkpoint"]),
                    attack_checkpoint="",
                    refine_checkpoint="",
                    arr_path="",
                    source_file=str(parsed["source_file"]),
                )
            )

    rows.sort(key=lambda row: (row.dataset, row.attack, row.poisoned_rate or -1.0, row.source_file))
    return rows


def collect_refine_rows(experiment_root: Path, dataset_filter: Optional[str], latest_only: bool) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    refine_root = experiment_root / "refine"
    if not refine_root.exists():
        return rows

    for results_path in refine_root.glob("*/eval/*/results.txt"):
        eval_dir = results_path.parent
        if latest_only and "latest" not in eval_dir.name:
            continue
        parsed = parse_refine_results(results_path)
        if dataset_filter and parsed["dataset"] != dataset_filter:
            continue
        if parsed["clean_accuracy"] is None or parsed["poisoned_accuracy"] is None:
            continue
        rows.append(
            SummaryRow(
                kind="refine",
                dataset=str(parsed["dataset"]),
                attack=str(parsed["attack"]),
                poisoned_rate=parsed["poisoned_rate"],
                clean_metric=str(parsed["clean_metric"]),
                clean_accuracy=parsed["clean_accuracy"],
                clean_correct=parsed["clean_correct"],
                clean_total=parsed["clean_total"],
                poisoned_metric=str(parsed["poisoned_metric"]),
                poisoned_accuracy=parsed["poisoned_accuracy"],
                poisoned_correct=parsed["poisoned_correct"],
                poisoned_total=parsed["poisoned_total"],
                checkpoint="",
                attack_checkpoint=str(parsed["attack_checkpoint"]),
                refine_checkpoint=str(parsed["refine_checkpoint"]),
                arr_path=str(parsed["arr_path"]),
                source_file=str(parsed["source_file"]),
            )
        )

    rows.sort(key=lambda row: (row.dataset, row.attack, row.poisoned_rate or -1.0, row.source_file))
    return rows


def collect_rows(experiment_root: Path, dataset_filter: Optional[str], latest_only: bool) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    rows.extend(collect_benign_rows(experiment_root, dataset_filter))
    rows.extend(collect_attack_rows(experiment_root, dataset_filter))
    rows.extend(collect_refine_rows(experiment_root, dataset_filter, latest_only))
    return rows


def write_csv(rows: List[SummaryRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(SummaryRow)]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(rows: List[SummaryRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2), encoding="utf-8")


def row_mtime(row: SummaryRow) -> float:
    try:
        return Path(row.source_file).stat().st_mtime
    except OSError:
        return 0.0


def make_key(dataset: str, attack: str, poisoned_rate: Optional[float]) -> tuple[str, str, Optional[float]]:
    normalized_rate = None if poisoned_rate is None else round(float(poisoned_rate), 6)
    return (dataset.lower(), attack.lower(), normalized_rate)


def pick_latest_by_key(rows: List[SummaryRow]) -> Dict[tuple[str, str, Optional[float]], SummaryRow]:
    selected: Dict[tuple[str, str, Optional[float]], SummaryRow] = {}
    for row in rows:
        key = make_key(row.dataset, row.attack, row.poisoned_rate)
        current = selected.get(key)
        if current is None or row_mtime(row) >= row_mtime(current):
            selected[key] = row
    return selected


def build_compare_rows(rows: List[SummaryRow]) -> List[CompareRow]:
    attack_rows = [row for row in rows if row.kind == "attack"]
    refine_rows = [row for row in rows if row.kind == "refine"]

    attack_latest = pick_latest_by_key(attack_rows)
    compare_rows: List[CompareRow] = []

    for refine in refine_rows:
        key = make_key(refine.dataset, refine.attack, refine.poisoned_rate)
        attack = attack_latest.get(key)

        clean_before = attack.clean_accuracy if attack else None
        asr_before = attack.poisoned_accuracy if attack else None
        ba_after = refine.clean_accuracy
        asr_after = refine.poisoned_accuracy

        delta_clean = None
        if clean_before is not None and ba_after is not None:
            delta_clean = ba_after - clean_before

        delta_asr = None
        if asr_before is not None and asr_after is not None:
            delta_asr = asr_after - asr_before

        compare_rows.append(
            CompareRow(
                dataset=refine.dataset,
                attack=refine.attack,
                poisoned_rate=refine.poisoned_rate,
                clean_before=clean_before,
                asr_before=asr_before,
                ba_after=ba_after,
                asr_after=asr_after,
                delta_clean=delta_clean,
                delta_asr=delta_asr,
                attack_source="" if attack is None else attack.source_file,
                refine_source=refine.source_file,
            )
        )

    compare_rows.sort(key=lambda row: (row.dataset, row.attack, row.poisoned_rate if row.poisoned_rate is not None else -1.0))
    return compare_rows


def write_compare_csv(rows: List[CompareRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(CompareRow)]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_compare_markdown(rows: List[CompareRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = ["# REFINE Before vs After", ""]
    if not rows:
        lines.append("No records found.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    columns = [
        "dataset",
        "attack",
        "poisoned_rate",
        "clean_before",
        "asr_before",
        "ba_after",
        "asr_after",
        "delta_clean",
        "delta_asr",
    ]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = [
            cell(row.dataset),
            cell(row.attack),
            cell(row.poisoned_rate),
            cell(row.clean_before),
            cell(row.asr_before),
            cell(row.ba_after),
            cell(row.asr_after),
            cell(row.delta_clean),
            cell(row.delta_asr),
        ]
        lines.append("| " + " | ".join(values) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cell(value: object, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format_float(value, digits=digits)
    return str(value)


def section_to_markdown(title: str, rows: List[SummaryRow], columns: List[str]) -> List[str]:
    lines = [f"## {title}"]
    if not rows:
        lines.append("No records found.")
        return lines

    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        payload = asdict(row)
        values = [cell(payload.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_markdown(rows: List[SummaryRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    benign_rows = [row for row in rows if row.kind == "benign"]
    attack_rows = [row for row in rows if row.kind == "attack"]
    refine_rows = [row for row in rows if row.kind == "refine"]

    lines: List[str] = ["# GTSRB Summary", ""]
    lines.extend(
        section_to_markdown(
            "Benign",
            benign_rows,
            ["dataset", "attack", "clean_accuracy", "clean_correct", "clean_total", "poisoned_metric", "poisoned_accuracy", "poisoned_correct", "poisoned_total", "checkpoint", "source_file"],
        )
    )
    lines.append("")
    lines.extend(
        section_to_markdown(
            "Attack",
            attack_rows,
            ["dataset", "attack", "poisoned_rate", "clean_accuracy", "clean_correct", "clean_total", "poisoned_metric", "poisoned_accuracy", "poisoned_correct", "poisoned_total", "checkpoint", "source_file"],
        )
    )
    lines.append("")
    lines.extend(
        section_to_markdown(
            "REFINE",
            refine_rows,
            ["dataset", "attack", "poisoned_rate", "clean_accuracy", "clean_correct", "clean_total", "poisoned_metric", "poisoned_accuracy", "poisoned_correct", "poisoned_total", "attack_checkpoint", "refine_checkpoint", "arr_path", "source_file"],
        )
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GTSRB experiment results into summary tables.")
    parser.add_argument("--experiment-root", default="experiments", help="Root directory of experiment outputs.")
    parser.add_argument("--dataset", default="gtsrb", help="Optional dataset filter, e.g. gtsrb or cub200.")
    parser.add_argument("--latest-only", action="store_true", help="Only include REFINE eval folders with 'latest' in their name.")
    parser.add_argument("--output", default="experiments/final_summary/gtsrb_summary.md", help="Markdown summary output path.")
    parser.add_argument("--csv-output", default="experiments/final_summary/gtsrb_summary.csv", help="CSV summary output path.")
    parser.add_argument("--json-output", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--compare-output", default="experiments/final_summary/gtsrb_refine_compare.md", help="Markdown compare output path.")
    parser.add_argument("--compare-csv-output", default="experiments/final_summary/gtsrb_refine_compare.csv", help="CSV compare output path.")
    args = parser.parse_args()

    experiment_root = to_path(args.experiment_root)
    dataset_filter = args.dataset.strip().lower() if args.dataset else None
    rows = collect_rows(experiment_root, dataset_filter, args.latest_only)

    markdown_path = to_path(args.output)
    csv_path = to_path(args.csv_output)
    write_markdown(rows, markdown_path)
    write_csv(rows, csv_path)
    if args.json_output:
        write_json(rows, to_path(args.json_output))

    compare_rows = build_compare_rows(rows)
    compare_markdown_path = to_path(args.compare_output)
    compare_csv_path = to_path(args.compare_csv_output)
    write_compare_markdown(compare_rows, compare_markdown_path)
    write_compare_csv(compare_rows, compare_csv_path)

    print(f"Collected {len(rows)} records -> {markdown_path}")
    print(f"CSV summary -> {csv_path}")
    print(f"Compare markdown -> {compare_markdown_path}")
    print(f"Compare CSV -> {compare_csv_path}")


if __name__ == "__main__":
    main()
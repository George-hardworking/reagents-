###################################################################################
# File: merge_results_chunks.py
# File Manager: Kaibiao ZHU
# Description: This file is to merge the chunked result files (block_results_*.json) into larger block_results files, each containing a specified number of records, sorted by their numeric IDs.
####################################################################################
"""Merge per-chunk result JSON files into larger block_results files."""

import re
import sys
import json
import argparse

from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, DefaultDict


def _extract_numeric_id(key: str) -> int:
    match = re.search(r"(\d+)", key)
    return int(match.group(1)) if match else -1


def _chunk_file_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"(\d+)-(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return sys.maxsize, path.name


def _load_chunk(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(
                f"Chunk file {path} must contain an object at the top level"
            )
        return data


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_all_json_files(base_dir: Path) -> DefaultDict[str, list]:
    file_map: DefaultDict[str, list] = defaultdict(list)
    for chunk_dir in base_dir.iterdir():
        if chunk_dir.is_dir():
            for fname in chunk_dir.iterdir():
                if fname.name.startswith("block_results_") and fname.name.endswith(
                    ".json"
                ):
                    file_map[fname.name].append(fname)
    return file_map


def merge_and_chunk_files(
    file_map: DefaultDict[str, list], output_dir: Path, block_size: int, overwrite: bool
):
    _ensure_output_dir(output_dir)
    existing_blocks = list(output_dir.glob("block_results_*.json"))
    if existing_blocks and not overwrite:
        raise RuntimeError(
            f"Output directory {output_dir} already contains block_results files. "
            "Use --overwrite to replace them."
        )
    for path in existing_blocks:
        path.unlink()

    for fname, files in file_map.items():
        combined: Dict[str, dict] = {}
        for f in files:
            chunk_data = _load_chunk(f)
            combined.update(chunk_data)
        if not combined:
            print(f"No records loaded from {fname}")
            continue
        ordered_items = sorted(
            combined.items(), key=lambda item: (_extract_numeric_id(item[0]), item[0])
        )
        total = len(ordered_items)
        blocks_written = 0
        for index in range(0, total, block_size):
            block_slice = ordered_items[index : index + block_size]
            start_id = _extract_numeric_id(block_slice[0][0])
            end_id = _extract_numeric_id(block_slice[-1][0])
            block_dict = {key: value for key, value in block_slice}
            output_path = output_dir / f"{fname.split('.')[0]}_{start_id}-{end_id}.json"
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(block_dict, handle, ensure_ascii=False, indent=2)
            blocks_written += 1
        print(
            f"Merged {len(files)} files for {fname} into {blocks_written} block files, total records: {total}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge chunked result files into block_results JSON files."
    )
    parser.add_argument(
        "--results-dir",
        default="results/2_single_math_qwen/merged",
        help="Directory for final merged block_results_*.json files",
    )  # 合并结果输出目录
    parser.add_argument(
        "--chunk-dir",
        help="Directory containing chunked block_results files (default: results/2_single_math_qwen/chunks)",
    )  # 所有chunk子文件夹的父目录
    parser.add_argument(
        "--block-size",
        type=int,
        default=1000,
        help="Number of records per merged block",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing block_results files in the output directory",
    )

    args = parser.parse_args()

    # 路径设置
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "results/13_singleself_gsm8k_dou/merged"
    chunk_dir = project_root / "results/13_singleself_gsm8k_dou/chunks"

    if not chunk_dir.exists() or not chunk_dir.is_dir():
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    file_map = find_all_json_files(chunk_dir)
    if not file_map:
        print(f"No block_results_*.json files found under {chunk_dir}")
        return
    # 参数设置
    block_size = 1000  # 每个块包含1000条记录
    overwrite = True  # 允许覆盖现有文件

    merge_and_chunk_files(file_map, output_dir, block_size, overwrite)


if __name__ == "__main__":
    main()

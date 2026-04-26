from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from ace_pre.data_build.tensor_builder import RelationTensorBuilder



@dataclass
class A1Sample:
    sample_id: str
    sequence: str
    length: int
    label_pIC50: float
    label_ic50_uM: float
    sample_weight: float
    task_role: str


class A1SequenceDataset(Dataset):
    """
    直接读取 A1 model_input 表，并调用之前写好的 RelationTensorBuilder
    构造关系张量。
    """

    def __init__(
        self,
        csv_path: str | Path,
        tensor_builder: RelationTensorBuilder,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.tensor_builder = tensor_builder

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        required_cols = [
            "sample_id",
            "sequence",
            "length",
            "label_pIC50",
            "label_ic50_uM",
            "sample_weight",
            "task_role",
        ]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}")

        self.df["sequence"] = self.df["sequence"].astype(str).str.upper().str.strip()
        self.df["length"] = self.df["length"].astype(int)

    def __len__(self) -> int:
        return len(self.df)

    def _row_to_sample(self, row: pd.Series) -> A1Sample:
        return A1Sample(
            sample_id=str(row["sample_id"]),
            sequence=str(row["sequence"]),
            length=int(row["length"]),
            label_pIC50=float(row["label_pIC50"]),
            label_ic50_uM=float(row["label_ic50_uM"]),
            sample_weight=float(row["sample_weight"]),
            task_role=str(row["task_role"]),
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        sample = self._row_to_sample(row)

        tensor_out = self.tensor_builder.build_single(sample.sequence)

        return {
            "sample_id": sample.sample_id,
            "sequence": sample.sequence,
            "length": sample.length,
            "task_role": sample.task_role,

            "x_hand": tensor_out.x_hand,             # [C,5,5]
            "pair_mask": tensor_out.pair_mask,       # [1,5,5]
            "residue_mask": tensor_out.residue_mask, # [5]
            "residue_ids": tensor_out.residue_ids,   # [5]

            "label_pIC50": torch.tensor(sample.label_pIC50, dtype=torch.float32),
            "label_ic50_uM": torch.tensor(sample.label_ic50_uM, dtype=torch.float32),
            "sample_weight": torch.tensor(sample.sample_weight, dtype=torch.float32),
        }


def a1_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "sequence": [item["sequence"] for item in batch],
        "length": torch.tensor([item["length"] for item in batch], dtype=torch.long),
        "task_role": [item["task_role"] for item in batch],

        "x_hand": torch.stack([item["x_hand"] for item in batch], dim=0),               # [B,C,5,5]
        "pair_mask": torch.stack([item["pair_mask"] for item in batch], dim=0),         # [B,1,5,5]
        "residue_mask": torch.stack([item["residue_mask"] for item in batch], dim=0),   # [B,5]
        "residue_ids": torch.stack([item["residue_ids"] for item in batch], dim=0),     # [B,5]

        "label_pIC50": torch.stack([item["label_pIC50"] for item in batch], dim=0).unsqueeze(-1),   # [B,1]
        "label_ic50_uM": torch.stack([item["label_ic50_uM"] for item in batch], dim=0).unsqueeze(-1),
        "sample_weight": torch.stack([item["sample_weight"] for item in batch], dim=0),              # [B]
    }
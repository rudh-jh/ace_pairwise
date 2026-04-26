from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ace_pre.data_build.tensor_builder import RelationTensorBuilder


def flatten_tensor_feature_dict(sequence: str, builder: RelationTensorBuilder) -> dict[str, float]:
    """
    使用 A1 的 handcrafted relation tensor 作为特征来源，
    将 [C, 5, 5] 展平为一维表格特征。
    """
    out = builder.build_single(sequence)
    x = out.x_hand.numpy()  # [C, 5, 5]

    feat = {}
    c_names = out.channel_names

    for c_idx, c_name in enumerate(c_names):
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                feat[f"{c_name}__r{i}__c{j}"] = float(x[c_idx, i, j])

    # 额外补几个很常用的显式字段，便于树模型利用
    feat["seq_length"] = float(out.length)
    for i in range(len(out.residue_ids)):
        feat[f"residue_id_{i}"] = int(out.residue_ids[i].item())

    return feat


def build_a1_flat_feature_frame(df: pd.DataFrame, sequence_col: str = "sequence") -> pd.DataFrame:
    builder = RelationTensorBuilder(max_len=5)
    rows = []

    for _, row in df.iterrows():
        seq = str(row[sequence_col]).strip().upper()
        rows.append(flatten_tensor_feature_dict(seq, builder))

    return pd.DataFrame(rows, index=df.index)
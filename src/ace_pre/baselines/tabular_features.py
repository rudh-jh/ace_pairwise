from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ace_pre.data_build.amino_acid_properties import build_default_property_table


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _aac_features(seq: str) -> dict[str, float]:
    seq = seq.strip().upper()
    length = len(seq)
    counts = {f"aac_{aa}": 0.0 for aa in AA_ORDER}
    if length == 0:
        return counts

    for aa in seq:
        if aa in counts:
            counts[f"aac_{aa}"] += 1.0

    for aa in AA_ORDER:
        counts[f"aac_{aa}"] /= length
    return counts


def _terminal_onehot_features(seq: str) -> dict[str, float]:
    seq = seq.strip().upper()
    feats = {}

    for aa in AA_ORDER:
        feats[f"nterm_{aa}"] = 1.0 if len(seq) >= 1 and seq[0] == aa else 0.0
        feats[f"cterm_{aa}"] = 1.0 if len(seq) >= 1 and seq[-1] == aa else 0.0

    return feats


def _global_physchem_features(seq: str) -> dict[str, float]:
    table = build_default_property_table()
    seq = seq.strip().upper()

    hyd = []
    pol = []
    chg = []
    aro = []
    vol = []
    donor = []
    acceptor = []

    pro_count = 0
    gly_count = 0
    branched_count = 0
    aromatic_count = 0
    hydrophobic_count = 0
    bulky_count = 0

    for aa in seq:
        p = table.get(aa)
        hyd.append(_safe_float(p["hydrophobicity"]))
        pol.append(_safe_float(p["polarity"]))
        chg.append(_safe_float(p["charge"]))
        aro.append(_safe_float(p["aromaticity"]))
        vol.append(_safe_float(p["volume"]))
        donor.append(_safe_float(p["hbond_donor"]))
        acceptor.append(_safe_float(p["hbond_acceptor"]))

        pro_count += int(_safe_float(p["pro_flag"]))
        gly_count += int(_safe_float(p["gly_flag"]))
        branched_count += int(_safe_float(p["branched_chain_flag"]))
        aromatic_count += int(_safe_float(p["aromatic_residue_flag"]))
        hydrophobic_count += int(_safe_float(p["is_hydrophobic_flag"]))
        bulky_count += int(_safe_float(p["is_bulky_flag"]))

    length = max(1, len(seq))

    feats = {
        "length": float(len(seq)),
        "hyd_mean": float(np.mean(hyd)),
        "hyd_std": float(np.std(hyd)),
        "pol_mean": float(np.mean(pol)),
        "pol_std": float(np.std(pol)),
        "chg_sum": float(np.sum(chg)),
        "chg_mean": float(np.mean(chg)),
        "aro_sum": float(np.sum(aro)),
        "vol_mean": float(np.mean(vol)),
        "vol_std": float(np.std(vol)),
        "donor_sum": float(np.sum(donor)),
        "acceptor_sum": float(np.sum(acceptor)),
        "pro_frac": pro_count / length,
        "gly_frac": gly_count / length,
        "branched_frac": branched_count / length,
        "aromatic_frac": aromatic_count / length,
        "hydrophobic_frac": hydrophobic_count / length,
        "bulky_frac": bulky_count / length,
    }

    if len(seq) >= 2:
        pair_hyd_diff = []
        pair_pol_diff = []
        pair_vol_diff = []
        pair_charge_prod = []

        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                p_i = table.get(seq[i])
                p_j = table.get(seq[j])
                pair_hyd_diff.append(abs(_safe_float(p_i["hydrophobicity"]) - _safe_float(p_j["hydrophobicity"])))
                pair_pol_diff.append(abs(_safe_float(p_i["polarity"]) - _safe_float(p_j["polarity"])))
                pair_vol_diff.append(abs(_safe_float(p_i["volume"]) - _safe_float(p_j["volume"])))
                pair_charge_prod.append(_safe_float(p_i["charge"]) * _safe_float(p_j["charge"]))

        feats.update({
            "pair_hyd_diff_mean": float(np.mean(pair_hyd_diff)),
            "pair_pol_diff_mean": float(np.mean(pair_pol_diff)),
            "pair_vol_diff_mean": float(np.mean(pair_vol_diff)),
            "pair_charge_prod_mean": float(np.mean(pair_charge_prod)),
        })
    else:
        feats.update({
            "pair_hyd_diff_mean": 0.0,
            "pair_pol_diff_mean": 0.0,
            "pair_vol_diff_mean": 0.0,
            "pair_charge_prod_mean": 0.0,
        })

    return feats


def sequence_to_descriptor_dict(seq: str) -> dict[str, float]:
    seq = seq.strip().upper()
    feats = {}
    feats.update(_aac_features(seq))
    feats.update(_terminal_onehot_features(seq))
    feats.update(_global_physchem_features(seq))
    return feats


def build_descriptor_frame(df: pd.DataFrame, sequence_col: str = "sequence") -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        seq = str(row[sequence_col]).strip().upper()
        rows.append(sequence_to_descriptor_dict(seq))
    return pd.DataFrame(rows, index=df.index)
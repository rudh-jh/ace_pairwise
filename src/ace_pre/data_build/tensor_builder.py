from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from ace_pre.data_build.amino_acid_properties import ResiduePropertyTable, build_default_property_table
from ace_pre.data_build.masks import apply_2d_mask, build_pair_mask, build_residue_mask


@dataclass
class RelationTensorOutput:
    sequence: str
    length: int
    x_hand: torch.Tensor
    pair_mask: torch.Tensor
    residue_mask: torch.Tensor
    residue_ids: torch.Tensor
    channel_names: List[str]


class RelationTensorBuilder:
    def __init__(
        self,
        property_table: ResiduePropertyTable | None = None,
        max_len: int = 5,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.property_table = property_table or build_default_property_table()
        self.max_len = max_len
        self.dtype = dtype
        self.channel_names = self._build_channel_names()
        self.channel_to_idx = {name: i for i, name in enumerate(self.channel_names)}

    def _build_channel_names(self) -> List[str]:
        return [
            "diag_hydrophobicity",
            "diag_polarity",
            "diag_charge",
            "diag_aromaticity",
            "diag_volume",
            "diag_hbond_donor",
            "diag_hbond_acceptor",
            "diag_pro_flag",
            "diag_gly_flag",
            "diag_branched_chain_flag",
            "diag_aromatic_residue_flag",
            "diag_is_n_terminal",
            "diag_is_c_terminal",
            "diag_is_internal",
            "diag_valid_flag",
            "pair_hydrophobicity_sum",
            "pair_hydrophobicity_abs_diff",
            "pair_hydrophobicity_signed_diff",
            "pair_charge_product",
            "pair_charge_signed_diff",
            "pair_polarity_abs_diff",
            "pair_volume_abs_diff",
            "pair_both_hydrophobic_flag",
            "pair_both_bulky_flag",
            "pair_aromatic_pair_flag",
            "pair_donor_acceptor_compatibility",
            "pair_terminal_interaction_flag",
            "pair_end_to_end_flag",
            "pair_end_to_internal_flag",
            "pair_forward_order_flag",
            "pair_reverse_order_flag",
            "pair_substitution_similarity_proxy",
            "pair_residue_class_compatibility",
            "heur_c_terminal_favorable_flag",
            "heur_n_terminal_favorable_flag",
            "heur_terminal_hydrophobic_pattern_flag",
        ]

    def num_channels(self) -> int:
        return len(self.channel_names)

    def _validate_sequence(self, seq: str) -> str:
        seq = seq.strip().upper()
        if len(seq) < 2 or len(seq) > self.max_len:
            raise ValueError(f"Sequence length must be in [2, {self.max_len}], got '{seq}' (len={len(seq)})")
        unknown = [aa for aa in seq if not self.property_table.has_residue(aa)]
        if unknown:
            raise ValueError(f"Sequence contains unknown residues: {unknown}, seq={seq}")
        return seq

    def _role_flags(self, position: int, length: int) -> Dict[str, float]:
        is_n = 1.0 if position == 0 else 0.0
        is_c = 1.0 if position == length - 1 else 0.0
        is_internal = 1.0 if (position > 0 and position < length - 1) else 0.0
        return {
            "is_n_terminal": is_n,
            "is_c_terminal": is_c,
            "is_internal": is_internal,
        }

    def _terminal_related_flags(self, i: int, j: int, length: int) -> Dict[str, float]:
        i_role = self._role_flags(i, length)
        j_role = self._role_flags(j, length)

        i_terminal = 1.0 if (i_role["is_n_terminal"] or i_role["is_c_terminal"]) else 0.0
        j_terminal = 1.0 if (j_role["is_n_terminal"] or j_role["is_c_terminal"]) else 0.0
        i_internal = i_role["is_internal"]
        j_internal = j_role["is_internal"]

        terminal_interaction = 1.0 if (i_terminal or j_terminal) else 0.0
        end_to_end = 1.0 if (i_terminal and j_terminal) else 0.0
        end_to_internal = 1.0 if ((i_terminal and j_internal) or (j_terminal and i_internal)) else 0.0

        return {
            "terminal_interaction": terminal_interaction,
            "end_to_end": end_to_end,
            "end_to_internal": end_to_internal,
        }

    def _substitution_similarity_proxy(self, aa_i: str, aa_j: str) -> float:
        if aa_i == aa_j:
            return 1.0

        class_i = self.property_table.class_name(aa_i)
        class_j = self.property_table.class_name(aa_j)

        if class_i == class_j:
            return 0.75

        acidic_basic = {class_i, class_j} == {"acidic", "basic"}
        if acidic_basic:
            return 0.30

        both_polar_family = class_i in {"polar", "amide"} and class_j in {"polar", "amide"}
        if both_polar_family:
            return 0.50

        both_hydrophobic_family = class_i in {"aliphatic", "aromatic", "sulfur", "special"} and \
                                  class_j in {"aliphatic", "aromatic", "sulfur", "special"}
        if both_hydrophobic_family:
            return 0.40

        return 0.0

    def _residue_class_compatibility(self, aa_i: str, aa_j: str) -> float:
        class_i = self.property_table.class_name(aa_i)
        class_j = self.property_table.class_name(aa_j)

        if class_i == class_j:
            return 1.0
        if {class_i, class_j} == {"acidic", "basic"}:
            return 0.8
        if class_i in {"polar", "amide"} and class_j in {"polar", "amide"}:
            return 0.7
        if class_i in {"aliphatic", "aromatic", "sulfur", "special"} and \
           class_j in {"aliphatic", "aromatic", "sulfur", "special"}:
            return 0.6
        return 0.0

    def _ace_heuristics(self, seq: str, i: int, j: int) -> Dict[str, float]:
        length = len(seq)
        c_term_aa = seq[-1]
        n_term_aa = seq[0]

        c_favorable = 1.0 if c_term_aa in {"P", "F", "Y", "W", "L", "I", "V", "M"} else 0.0
        n_favorable = 1.0 if n_term_aa in {"V", "I", "L", "A", "R", "K", "F"} else 0.0

        i_role = self._role_flags(i, length)
        j_role = self._role_flags(j, length)

        touches_terminal = (
            i_role["is_n_terminal"] or i_role["is_c_terminal"] or
            j_role["is_n_terminal"] or j_role["is_c_terminal"]
        )

        prop_i = self.property_table.get(seq[i])
        prop_j = self.property_table.get(seq[j])

        terminal_hydrophobic_pattern = 1.0 if (
            touches_terminal and (
                float(prop_i["is_hydrophobic_flag"]) > 0.0 or
                float(prop_j["is_hydrophobic_flag"]) > 0.0
            )
        ) else 0.0

        heur_c = c_favorable if (i == length - 1 or j == length - 1) else 0.0
        heur_n = n_favorable if (i == 0 or j == 0) else 0.0

        return {
            "heur_c_terminal_favorable_flag": heur_c,
            "heur_n_terminal_favorable_flag": heur_n,
            "heur_terminal_hydrophobic_pattern_flag": terminal_hydrophobic_pattern,
        }

    def build_single(self, seq: str) -> RelationTensorOutput:
        seq = self._validate_sequence(seq)
        length = len(seq)

        x = torch.zeros(
            (self.num_channels(), self.max_len, self.max_len),
            dtype=self.dtype
        )

        residue_ids = torch.zeros(self.max_len, dtype=torch.long)
        residue_mask = build_residue_mask(length, max_len=self.max_len, dtype=self.dtype)
        pair_mask = build_pair_mask(length, max_len=self.max_len, dtype=self.dtype)

        for pos, aa in enumerate(seq):
            residue_ids[pos] = self.property_table.get_index(aa)

        for i, aa in enumerate(seq):
            prop_i = self.property_table.get(aa)
            role_i = self._role_flags(i, length)

            x[self.channel_to_idx["diag_hydrophobicity"], i, i] = float(prop_i["hydrophobicity"])
            x[self.channel_to_idx["diag_polarity"], i, i] = float(prop_i["polarity"])
            x[self.channel_to_idx["diag_charge"], i, i] = float(prop_i["charge"])
            x[self.channel_to_idx["diag_aromaticity"], i, i] = float(prop_i["aromaticity"])
            x[self.channel_to_idx["diag_volume"], i, i] = float(prop_i["volume"])
            x[self.channel_to_idx["diag_hbond_donor"], i, i] = float(prop_i["hbond_donor"])
            x[self.channel_to_idx["diag_hbond_acceptor"], i, i] = float(prop_i["hbond_acceptor"])
            x[self.channel_to_idx["diag_pro_flag"], i, i] = float(prop_i["pro_flag"])
            x[self.channel_to_idx["diag_gly_flag"], i, i] = float(prop_i["gly_flag"])
            x[self.channel_to_idx["diag_branched_chain_flag"], i, i] = float(prop_i["branched_chain_flag"])
            x[self.channel_to_idx["diag_aromatic_residue_flag"], i, i] = float(prop_i["aromatic_residue_flag"])
            x[self.channel_to_idx["diag_is_n_terminal"], i, i] = role_i["is_n_terminal"]
            x[self.channel_to_idx["diag_is_c_terminal"], i, i] = role_i["is_c_terminal"]
            x[self.channel_to_idx["diag_is_internal"], i, i] = role_i["is_internal"]
            x[self.channel_to_idx["diag_valid_flag"], i, i] = 1.0

        for i, aa_i in enumerate(seq):
            prop_i = self.property_table.get(aa_i)

            for j, aa_j in enumerate(seq):
                if i == j:
                    continue

                prop_j = self.property_table.get(aa_j)

                hyd_i = float(prop_i["hydrophobicity"])
                hyd_j = float(prop_j["hydrophobicity"])
                chg_i = float(prop_i["charge"])
                chg_j = float(prop_j["charge"])
                pol_i = float(prop_i["polarity"])
                pol_j = float(prop_j["polarity"])
                vol_i = float(prop_i["volume"])
                vol_j = float(prop_j["volume"])
                donor_i = float(prop_i["hbond_donor"])
                donor_j = float(prop_j["hbond_donor"])
                acceptor_i = float(prop_i["hbond_acceptor"])
                acceptor_j = float(prop_j["hbond_acceptor"])

                x[self.channel_to_idx["pair_hydrophobicity_sum"], i, j] = hyd_i + hyd_j
                x[self.channel_to_idx["pair_hydrophobicity_abs_diff"], i, j] = abs(hyd_i - hyd_j)
                x[self.channel_to_idx["pair_hydrophobicity_signed_diff"], i, j] = hyd_i - hyd_j
                x[self.channel_to_idx["pair_charge_product"], i, j] = chg_i * chg_j
                x[self.channel_to_idx["pair_charge_signed_diff"], i, j] = chg_i - chg_j
                x[self.channel_to_idx["pair_polarity_abs_diff"], i, j] = abs(pol_i - pol_j)
                x[self.channel_to_idx["pair_volume_abs_diff"], i, j] = abs(vol_i - vol_j)

                x[self.channel_to_idx["pair_both_hydrophobic_flag"], i, j] = \
                    1.0 if (float(prop_i["is_hydrophobic_flag"]) > 0.0 and float(prop_j["is_hydrophobic_flag"]) > 0.0) else 0.0

                x[self.channel_to_idx["pair_both_bulky_flag"], i, j] = \
                    1.0 if (float(prop_i["is_bulky_flag"]) > 0.0 and float(prop_j["is_bulky_flag"]) > 0.0) else 0.0

                x[self.channel_to_idx["pair_aromatic_pair_flag"], i, j] = \
                    1.0 if (float(prop_i["aromatic_residue_flag"]) > 0.0 and float(prop_j["aromatic_residue_flag"]) > 0.0) else 0.0

                donor_acceptor_compat = 1.0 if (
                    (donor_i > 0.0 and acceptor_j > 0.0) or
                    (acceptor_i > 0.0 and donor_j > 0.0)
                ) else 0.0
                x[self.channel_to_idx["pair_donor_acceptor_compatibility"], i, j] = donor_acceptor_compat

                term_flags = self._terminal_related_flags(i, j, length)
                x[self.channel_to_idx["pair_terminal_interaction_flag"], i, j] = term_flags["terminal_interaction"]
                x[self.channel_to_idx["pair_end_to_end_flag"], i, j] = term_flags["end_to_end"]
                x[self.channel_to_idx["pair_end_to_internal_flag"], i, j] = term_flags["end_to_internal"]

                x[self.channel_to_idx["pair_forward_order_flag"], i, j] = 1.0 if i < j else 0.0
                x[self.channel_to_idx["pair_reverse_order_flag"], i, j] = 1.0 if i > j else 0.0

                x[self.channel_to_idx["pair_substitution_similarity_proxy"], i, j] = \
                    self._substitution_similarity_proxy(aa_i, aa_j)

                x[self.channel_to_idx["pair_residue_class_compatibility"], i, j] = \
                    self._residue_class_compatibility(aa_i, aa_j)

                heur = self._ace_heuristics(seq, i, j)
                x[self.channel_to_idx["heur_c_terminal_favorable_flag"], i, j] = heur["heur_c_terminal_favorable_flag"]
                x[self.channel_to_idx["heur_n_terminal_favorable_flag"], i, j] = heur["heur_n_terminal_favorable_flag"]
                x[self.channel_to_idx["heur_terminal_hydrophobic_pattern_flag"], i, j] = \
                    heur["heur_terminal_hydrophobic_pattern_flag"]

        x = apply_2d_mask(x, pair_mask)

        return RelationTensorOutput(
            sequence=seq,
            length=length,
            x_hand=x,
            pair_mask=pair_mask,
            residue_mask=residue_mask,
            residue_ids=residue_ids,
            channel_names=list(self.channel_names),
        )

    def build_batch(self, seqs: Sequence[str]) -> Dict[str, torch.Tensor | List[str]]:
        outputs = [self.build_single(seq) for seq in seqs]

        x_hand = torch.stack([o.x_hand for o in outputs], dim=0)
        pair_mask = torch.stack([o.pair_mask for o in outputs], dim=0)
        residue_mask = torch.stack([o.residue_mask for o in outputs], dim=0)
        residue_ids = torch.stack([o.residue_ids for o in outputs], dim=0)
        lengths = torch.tensor([o.length for o in outputs], dtype=torch.long)

        return {
            "sequences": [o.sequence for o in outputs],
            "x_hand": x_hand,
            "pair_mask": pair_mask,
            "residue_mask": residue_mask,
            "residue_ids": residue_ids,
            "lengths": lengths,
            "channel_names": list(self.channel_names),
        }
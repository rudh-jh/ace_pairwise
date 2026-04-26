from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping


STANDARD_AA_ORDER: List[str] = list("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class ResidueProperties:
    hydrophobicity: float
    polarity: float
    charge: float
    aromaticity: float
    volume: float
    hbond_donor: float
    hbond_acceptor: float
    pro_flag: float
    gly_flag: float
    branched_chain_flag: float
    aromatic_residue_flag: float
    is_hydrophobic_flag: float
    is_bulky_flag: float
    residue_class: str


def _raw_property_table() -> Dict[str, ResidueProperties]:
    return {
        "A": ResidueProperties(1.8, 8.1, 0.0, 0.0, 88.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, "aliphatic"),
        "C": ResidueProperties(2.5, 5.5, 0.0, 0.0, 108.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, "sulfur"),
        "D": ResidueProperties(-3.5, 13.0, -1.0, 0.0, 111.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "acidic"),
        "E": ResidueProperties(-3.5, 12.3, -1.0, 0.0, 138.4, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "acidic"),
        "F": ResidueProperties(2.8, 5.2, 0.0, 1.0, 189.9, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, "aromatic"),
        "G": ResidueProperties(-0.4, 9.0, 0.0, 0.0, 60.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, "special"),
        "H": ResidueProperties(-3.2, 10.4, 0.1, 1.0, 153.2, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, "basic"),
        "I": ResidueProperties(4.5, 5.2, 0.0, 0.0, 166.7, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, "aliphatic"),
        "K": ResidueProperties(-3.9, 11.3, 1.0, 0.0, 168.6, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "basic"),
        "L": ResidueProperties(3.8, 4.9, 0.0, 0.0, 166.7, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, "aliphatic"),
        "M": ResidueProperties(1.9, 5.7, 0.0, 0.0, 162.9, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, "sulfur"),
        "N": ResidueProperties(-3.5, 11.6, 0.0, 0.0, 114.1, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "amide"),
        "P": ResidueProperties(-1.6, 8.0, 0.0, 0.0, 112.7, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, "special"),
        "Q": ResidueProperties(-3.5, 10.5, 0.0, 0.0, 143.8, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "amide"),
        "R": ResidueProperties(-4.5, 10.5, 1.0, 0.0, 173.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "basic"),
        "S": ResidueProperties(-0.8, 9.2, 0.0, 0.0, 89.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "polar"),
        "T": ResidueProperties(-0.7, 8.6, 0.0, 0.0, 116.1, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "polar"),
        "V": ResidueProperties(4.2, 5.9, 0.0, 0.0, 140.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, "aliphatic"),
        "W": ResidueProperties(-0.9, 5.4, 0.0, 1.0, 227.8, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, "aromatic"),
        "Y": ResidueProperties(-1.3, 6.2, 0.0, 1.0, 193.6, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, "aromatic"),
    }


def _min_max_normalize(values: Mapping[str, float]) -> Dict[str, float]:
    vmin = min(values.values())
    vmax = max(values.values())
    if vmax == vmin:
        return {k: 0.0 for k in values}
    return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}


class ResiduePropertyTable:
    def __init__(self) -> None:
        raw = _raw_property_table()

        hyd = _min_max_normalize({aa: p.hydrophobicity for aa, p in raw.items()})
        pol = _min_max_normalize({aa: p.polarity for aa, p in raw.items()})
        vol = _min_max_normalize({aa: p.volume for aa, p in raw.items()})
        donor = _min_max_normalize({aa: p.hbond_donor for aa, p in raw.items()})
        acceptor = _min_max_normalize({aa: p.hbond_acceptor for aa, p in raw.items()})

        self._table: Dict[str, Dict[str, float | str]] = {}
        for aa, p in raw.items():
            self._table[aa] = {
                "hydrophobicity": hyd[aa],
                "polarity": pol[aa],
                "charge": p.charge,
                "aromaticity": p.aromaticity,
                "volume": vol[aa],
                "hbond_donor": donor[aa],
                "hbond_acceptor": acceptor[aa],
                "pro_flag": p.pro_flag,
                "gly_flag": p.gly_flag,
                "branched_chain_flag": p.branched_chain_flag,
                "aromatic_residue_flag": p.aromatic_residue_flag,
                "is_hydrophobic_flag": p.is_hydrophobic_flag,
                "is_bulky_flag": p.is_bulky_flag,
                "residue_class": p.residue_class,
            }

        self._residue_to_index: Dict[str, int] = {"[PAD]": 0}
        for i, aa in enumerate(STANDARD_AA_ORDER, start=1):
            self._residue_to_index[aa] = i

    @property
    def residue_to_index(self) -> Dict[str, int]:
        return dict(self._residue_to_index)

    @property
    def index_to_residue(self) -> Dict[int, str]:
        return {v: k for k, v in self._residue_to_index.items()}

    @property
    def valid_residues(self) -> List[str]:
        return list(STANDARD_AA_ORDER)

    def has_residue(self, aa: str) -> bool:
        return aa in self._table

    def get(self, aa: str) -> Dict[str, float | str]:
        aa = aa.upper()
        if aa not in self._table:
            raise KeyError(f"Unknown amino acid: {aa}")
        return dict(self._table[aa])

    def get_index(self, aa: str) -> int:
        aa = aa.upper()
        if aa not in self._residue_to_index:
            raise KeyError(f"Unknown amino acid for index mapping: {aa}")
        return self._residue_to_index[aa]

    def class_name(self, aa: str) -> str:
        return str(self.get(aa)["residue_class"])

    def feature_names(self) -> List[str]:
        return [
            "hydrophobicity",
            "polarity",
            "charge",
            "aromaticity",
            "volume",
            "hbond_donor",
            "hbond_acceptor",
            "pro_flag",
            "gly_flag",
            "branched_chain_flag",
            "aromatic_residue_flag",
            "is_hydrophobic_flag",
            "is_bulky_flag",
        ]


def build_default_property_table() -> ResiduePropertyTable:
    return ResiduePropertyTable()


def build_residue_to_index() -> Dict[str, int]:
    return build_default_property_table().residue_to_index
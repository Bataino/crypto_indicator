"""Feature engineering: C, V, N, P → NSI, MREI, Rotation Gap."""

from cmram.features.compression import score_C
from cmram.features.gap import rotation_gap
from cmram.features.mrei import compute_mrei
from cmram.features.narrative import score_N
from cmram.features.nsi import compute_nsi
from cmram.features.pipeline import compute_features_daily, features_summary
from cmram.features.price import score_P
from cmram.features.volume import score_V

__all__ = [
    "score_C",
    "score_V",
    "score_N",
    "score_P",
    "compute_nsi",
    "compute_mrei",
    "rotation_gap",
    "compute_features_daily",
    "features_summary",
]

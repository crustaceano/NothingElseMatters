"""Lab 3: composite L1, Frank–Wolfe, barrier, and accelerated proximal methods."""

from .counted_oracle import (
    CountedBarrierOracle,
    CountedCompositeOracle,
    CountedNonsmoothOracle,
    CountedSmoothOracle,
)
from .experiment_utils import (
    radius_from_reference,
    synthetic_classification,
    synthetic_regression,
    tune_lambda_for_zero_fraction,
)
from .oracles import (
    BarrierL1Oracle,
    ClassificationNonsmoothOracle,
    ClassificationProxOracle,
    L1RegOracle,
    LogCoshL2Oracle,
    LogisticLossOracle,
    QuadraticOracle,
    RegressionNonsmoothOracle,
    RegressionProxOracle,
    RegressionSmoothOracle,
)
from .optimization import (
    away_step_frank_wolfe_method,
    barrier_method,
    frank_wolfe_method,
    proximal_fast_gradient_method,
    proximal_gradient_method,
    subgradient_method,
)
from .paths import figs_dir, project_root

__all__ = [
    "BarrierL1Oracle",
    "ClassificationNonsmoothOracle",
    "ClassificationProxOracle",
    "CountedBarrierOracle",
    "CountedCompositeOracle",
    "CountedNonsmoothOracle",
    "CountedSmoothOracle",
    "L1RegOracle",
    "LogCoshL2Oracle",
    "LogisticLossOracle",
    "QuadraticOracle",
    "RegressionNonsmoothOracle",
    "RegressionProxOracle",
    "RegressionSmoothOracle",
    "away_step_frank_wolfe_method",
    "barrier_method",
    "frank_wolfe_method",
    "proximal_fast_gradient_method",
    "proximal_gradient_method",
    "subgradient_method",
    "figs_dir",
    "project_root",
    "synthetic_regression",
    "synthetic_classification",
    "tune_lambda_for_zero_fraction",
    "radius_from_reference",
]

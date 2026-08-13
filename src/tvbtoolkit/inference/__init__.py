"""Simulation-based parameter inference for TVBToolkit whole-brain models."""

from .adex import AdExBOLDSimulator, extract_bold_monitor
from .brain_act import (
    BrainActBOLDRecord,
    aal90_interleaved_to_symmetric_index,
    load_brain_act_bold_mat,
)
from .features import BOLDFeatureConfig, BOLDFeatureExtractor
from .parameters import AdExParameterSpec, AdExPrior, make_sbi_prior
from .sbi import (
    SimulationDataset,
    sample_vbi_posterior,
    simulate_prior,
    train_vbi_posterior,
)

__all__ = [
    "AdExBOLDSimulator",
    "AdExParameterSpec",
    "AdExPrior",
    "BOLDFeatureConfig",
    "BOLDFeatureExtractor",
    "BrainActBOLDRecord",
    "SimulationDataset",
    "aal90_interleaved_to_symmetric_index",
    "extract_bold_monitor",
    "load_brain_act_bold_mat",
    "make_sbi_prior",
    "sample_vbi_posterior",
    "simulate_prior",
    "train_vbi_posterior",
]

"""Portable, frozen clinical v22 research inference package template."""

__all__ = ["ClinicalV22Predictor"]


def __getattr__(name):
    if name == "ClinicalV22Predictor":
        from .predict import ClinicalV22Predictor
        return ClinicalV22Predictor
    raise AttributeError(name)

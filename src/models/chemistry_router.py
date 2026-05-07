"""ChemistryRouter — deployable wrapper around the per-chemistry XGBoost
SoH submodels trained by `scripts/train_xgboost_per_chemistry.py`.

At inference time, given (X, chemistry) the router dispatches each row to
the appropriate per-chemistry submodel and returns SoH predictions. Cells
of an unknown chemistry fall back to the global audited model. Cell-by-cell
dispatch is vectorized: rows are grouped by chemistry, predicted in batch,
then re-assembled in the original order.

Per Iter-3 §3.10 results, this architecture lifts minority-chemistry grade
accuracy by 1–2 pp (NCA +1.96, Na-ion +1.45, LCO +1.20, "other" +0.79)
relative to the global model alone, with NMC and LFP tying with the global
(large samples already learnt).

Loading
-------
    from src.models.chemistry_router import ChemistryRouter
    router = ChemistryRouter.load("models/per_chemistry/router_manifest.json")

Inference
---------
    preds = router.predict_soh(X, chemistries)   # shape (n_rows,)
    grades = router.predict_grade(X, chemistries) # shape (n_rows,) — A/B/C/D
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import xgboost as xgb


@dataclass
class ChemistryRouter:
    """Loaded set of per-chemistry XGBoost models keyed by chemistry label."""
    models: dict                # chemistry -> xgb.XGBRegressor
    fallback_model: Optional[xgb.XGBRegressor]
    feature_names: list
    manifest: dict

    @classmethod
    def load(cls, manifest_path: str | Path) -> "ChemistryRouter":
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        # `model_path` entries in the manifest are project-root-relative
        # (e.g. "models/per_chemistry/NMC/xgboost_soh_audited.json"); resolve
        # them relative to manifest_path.parents[2] which is the project root.
        project_root = manifest_path.resolve().parents[2]

        models: dict = {}
        for entry in manifest["chemistries"]:
            m = xgb.XGBRegressor()
            m.load_model(str(project_root / entry["model_path"]))
            models[entry["chemistry"]] = m

        fallback = None
        if manifest.get("fallback_global_model"):
            fb_path = project_root / manifest["fallback_global_model"]
            if fb_path.exists():
                fallback = xgb.XGBRegressor()
                fallback.load_model(str(fb_path))

        feature_names_path = project_root / manifest["shared_feature_names"]
        feature_names = json.loads(feature_names_path.read_text())

        return cls(
            models=models,
            fallback_model=fallback,
            feature_names=feature_names,
            manifest=manifest,
        )

    def predict_soh(self, X: np.ndarray, chemistries: Iterable[str]) -> np.ndarray:
        """Per-row dispatch: group rows by chemistry, predict in batch, reassemble.

        Rows whose chemistry has no per-chemistry model fall back to the global
        audited model. If neither is available, raises KeyError.
        """
        chemistries = np.asarray(list(chemistries))
        if len(chemistries) != X.shape[0]:
            raise ValueError(
                f"chemistries length ({len(chemistries)}) must match X rows ({X.shape[0]})"
            )
        out = np.full(X.shape[0], np.nan, dtype=np.float32)
        for chem in np.unique(chemistries):
            mask = chemistries == chem
            model = self.models.get(chem) or self.fallback_model
            if model is None:
                raise KeyError(
                    f"No model for chemistry {chem!r} and no fallback available"
                )
            out[mask] = model.predict(X[mask]).astype(np.float32)
        return out

    def predict_grade(self, X: np.ndarray, chemistries: Iterable[str]) -> np.ndarray:
        """SoH→Grade dispatch via the same A/B/C/D thresholds used elsewhere
        (cf. `src.data.training_data.soh_to_grade`).

        Returns string array of grades.
        """
        soh = self.predict_soh(X, chemistries)
        return np.where(soh > 80, "A",
                np.where(soh > 60, "B",
                  np.where(soh > 40, "C", "D")))

    def known_chemistries(self) -> list[str]:
        return sorted(self.models.keys())

    def __repr__(self) -> str:
        return (f"ChemistryRouter(known={self.known_chemistries()}, "
                f"fallback={'yes' if self.fallback_model else 'no'}, "
                f"n_features={len(self.feature_names)})")

"""
Maps framework outputs to the unified Battery DPP schema.

Reconciles EU Regulation 2023/1542 Annex XIII, GBA Battery Pass Data Model
v1.2.0 (DIN DKE SPEC 99100), and India BWMR 2022/2024/2025.

Schema definition: data/processed/dpp/unified_dpp_schema.json
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    RESULTS_DIR,
    PROCESSED_DIR,
    BWMR_RECOVERY_TARGETS,
    BWMR_RECYCLED_CONTENT,
    GRADE_THRESHOLDS,
)

SCHEMA_VERSION = "1.0.0"
FRAMEWORK_NAME = "EV Battery EoL Routing Framework"
FRAMEWORK_VERSION = "0.9.0-stage12"
SCHEMA_PATH = PROCESSED_DIR / "dpp" / "unified_dpp_schema.json"

CHEMISTRY_CRM_MAP = {
    "NMC": ["Co", "Li", "Ni", "natural graphite"],
    "NCA": ["Co", "Li", "Ni", "natural graphite"],
    "LCO": ["Co", "Li", "natural graphite"],
    "LFP": ["Li", "natural graphite"],
    "LMO": ["Li", "natural graphite"],
    "Zn-ion": [],
    "Na-ion": ["natural graphite"],
}

ROUTE_FROM_GRADE = {
    "A": "Grid-scale ESS",
    "B": "Home/Distributed ESS",
    "C": "Component Reuse",
    "D": "Direct Recycling",
}


def grade_from_soh(soh_pct: float) -> str:
    if soh_pct > GRADE_THRESHOLDS["A"]:
        return "A"
    if soh_pct > GRADE_THRESHOLDS["B"]:
        return "B"
    if soh_pct > GRADE_THRESHOLDS["C"]:
        return "C"
    return "D"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_dpp(
    *,
    battery_id: str,
    chemistry: str,
    form_factor: str | None,
    nominal_Ah: float,
    voltage_min_V: float,
    voltage_max_V: float,
    cycles_completed: int,
    soh_percent: float,
    rul_remaining_cycles: float | None,
    estimation_method: str,
    estimation_confidence: dict,
    data_source: str,
    grade: str | None = None,
    recommended_route: str | None = None,
    route_score: float | None = None,
    mcdm_criteria: list | None = None,
    mcdm_weights: dict | None = None,
    all_route_scores: list | None = None,
    route_ranking_method: str = "Fuzzy BWM-TOPSIS",
    second_life: bool = False,
    data_sources: list | None = None,
    model_artifacts: list | None = None,
    carbon_footprint_kgCO2eq_per_kWh: float | None = None,
    carbon_status: str = "placeholder",
) -> dict:
    """
    Construct a unified-schema-compliant DPP for a single battery.

    Required inputs come from unified.parquet + ML inference + MCDM routing.
    Fields not yet wired (carbon, supply chain) emit literature-default or
    placeholder objects with `calculation_status` flagged accordingly.
    """
    grade = grade or grade_from_soh(soh_percent)
    recommended_route = recommended_route or ROUTE_FROM_GRADE[grade]
    chem_upper = (chemistry or "other").upper().replace("-ION", "-ion")

    product_status = "re-used" if second_life else "original"
    if grade == "D":
        product_status = "waste"

    bwmr_recovery_fy, bwmr_recovery_pct = max(BWMR_RECOVERY_TARGETS.items(), key=lambda kv: kv[1])
    bwmr_rc_year, bwmr_rc_pct = max(BWMR_RECYCLED_CONTENT.items(), key=lambda kv: kv[1])

    dpp = {
        "dpp_meta": {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _now_iso(),
            "framework": FRAMEWORK_NAME,
            "framework_version": FRAMEWORK_VERSION,
            "regulatory_alignment": [
                "EU Regulation 2023/1542 Annex XIII",
                "GBA Battery Pass Data Model v1.2.0 (DIN DKE SPEC 99100:2025-02)",
                "India BWMR 2022 + 2024 amendments + 2025 amendment",
            ],
        },

        "identity": {
            "dpp_id": f"DPP-{uuid.uuid4()}",
            "battery_id": battery_id,
            "product_status": product_status,
            "manufacturer_name": None,
            "manufacturer_country": None,
            "production_date": None,
            "placed_on_market_date": None,
            "form_factor": form_factor,
        },

        "chemistry_and_composition": {
            "cathode_chemistry": chem_upper,
            "anode_chemistry": "graphite" if chem_upper not in ("Zn-ion",) else None,
            "electrolyte_type": None,
            "critical_raw_materials_present": CHEMISTRY_CRM_MAP.get(chem_upper, []),
            "hazardous_substances": {
                "mercury": False,
                "cadmium": False,
                "lead": chem_upper == "Lead-acid",
            },
        },

        "performance_and_durability": {
            "rated_capacity_Ah": float(nominal_Ah),
            "nominal_voltage_V": (voltage_min_V + voltage_max_V) / 2,
            "voltage_min_V": float(voltage_min_V),
            "voltage_max_V": float(voltage_max_V),
            "operating_temp_min_C": -10.0,
            "operating_temp_max_C": 60.0,
            "expected_lifetime_cycles": int(cycles_completed + (rul_remaining_cycles or 0)),
            "capacity_threshold_for_exhaustion_pct": 80.0,
            "c_rate_relevant_test": None,
            "initial_round_trip_efficiency": None,
            "internal_resistance_ohm": None,
        },

        "state_of_health": {
            "soh_percent": round(float(soh_percent), 2),
            "soh_at_market_percent": 100.0,
            "rul_remaining_cycles": round(float(rul_remaining_cycles), 0) if rul_remaining_cycles is not None else None,
            "cycles_completed": int(cycles_completed),
            "estimation_method": estimation_method,
            "estimation_confidence": estimation_confidence,
            "last_assessed": _now_iso(),
            "data_source": data_source,
            "negative_events_log": [],
        },

        "carbon_footprint": {
            "carbon_footprint_kgCO2eq_per_kWh": carbon_footprint_kgCO2eq_per_kWh,
            "performance_class": None,
            "lifecycle_stage_breakdown_kgCO2eq_per_kWh": {
                "manufacturing": None,
                "distribution": None,
                "use_phase": None,
                "end_of_life": None,
            },
            "allocation_method": None,
            "calculation_status": carbon_status,
        },

        "supply_chain_due_diligence": {
            "policy_url": None,
            "cobalt_origin": [],
            "lithium_origin": [],
            "nickel_origin": [],
            "graphite_origin": [],
            "third_party_audit_report_id": None,
            "calculation_status": "placeholder",
        },

        "circularity_and_eol": {
            "recycled_content_share": {
                "cobalt": None,
                "lithium": None,
                "nickel": None,
                "lead": None,
            },
            "renewable_content_share": None,
            "grade": grade,
            "recommended_route": recommended_route,
            "route_score": float(route_score) if route_score is not None else None,
            "route_ranking_method": route_ranking_method,
            "mcdm_criteria": mcdm_criteria or ["SoH", "Value", "Carbon", "Compliance", "Safety", "EPR Return"],
            "mcdm_weights": mcdm_weights or {},
            "all_route_scores": all_route_scores or [],
            "sensitivity_robustness": {
                "spearman_rho": None,
                "scenarios_evaluated": None,
            },
            "epr_compliance": {
                "obligated_party": None,
                "take_back_route": "Authorised recycler" if grade in ("C", "D") else "Refurbisher",
                "recovery_target_fy": bwmr_recovery_fy,
                "recovery_target_pct": float(bwmr_recovery_pct),
                "recycled_content_target_year": bwmr_rc_year,
                "recycled_content_target_pct": float(bwmr_rc_pct),
                "epr_certificate_id": None,
            },
        },

        "labels_and_compliance": {
            "separate_collection_symbol_present": True,
            "heavy_metal_marking": None,
            "eu_declaration_of_conformity_id": None,
            "bwmr_compliance_form": "BWMR Form 6 (EPR Plan)",
            "ce_marking": False,
        },

        "dismantling_and_safety": {
            "dismantling_diagram_url": None,
            "disassembly_sequence_url": None,
            "tools_required": [],
            "access_level": "legitimate_interest",
        },

        "provenance": {
            "data_sources": data_sources or [],
            "model_artifacts": model_artifacts or [],
            "schema_validated": False,
            "coverage_pct": None,
        },
    }

    coverage = compute_dpp_coverage(dpp)
    dpp["provenance"]["coverage_pct"] = coverage
    return dpp


def validate_against_schema(dpp: dict) -> tuple[bool, list]:
    """Validate a DPP dict against the unified schema. Returns (ok, errors)."""
    try:
        import jsonschema
    except ImportError:
        return False, ["jsonschema not installed; run `pip install jsonschema`"]

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(dpp), key=lambda e: e.path)
    if not errors:
        return True, []
    return False, [f"{list(e.path)}: {e.message}" for e in errors]


def compute_dpp_coverage(dpp: dict) -> float:
    """
    Coverage = fraction of schema fields populated with non-null, substantive
    values. Counts leaves (not nested objects) under the 9 categories the
    schema defines, weighted equally per category to avoid the carbon/supply-
    chain placeholders dragging the headline number disproportionately.
    """

    def has_value(v):
        if v is None:
            return False
        if isinstance(v, str):
            return v != "" and v != "placeholder"
        if isinstance(v, list):
            return len(v) > 0
        if isinstance(v, dict):
            return any(has_value(x) for x in v.values())
        return True

    categories = [
        "identity",
        "chemistry_and_composition",
        "performance_and_durability",
        "state_of_health",
        "carbon_footprint",
        "supply_chain_due_diligence",
        "circularity_and_eol",
        "labels_and_compliance",
        "dismantling_and_safety",
    ]

    per_cat_scores = []
    for cat in categories:
        block = dpp.get(cat, {})
        if not isinstance(block, dict) or not block:
            per_cat_scores.append(0.0)
            continue
        leaves = list(block.values())
        if not leaves:
            per_cat_scores.append(0.0)
            continue
        populated = sum(1 for v in leaves if has_value(v))
        per_cat_scores.append(populated / len(leaves))

    return round(sum(per_cat_scores) / len(per_cat_scores), 3)


def save_dpp(dpp: dict, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or RESULTS_DIR / "dpp_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    battery_id = dpp["identity"]["battery_id"].replace("/", "_")
    filepath = output_dir / f"dpp_{battery_id}.json"
    with open(filepath, "w") as f:
        json.dump(dpp, f, indent=2, default=str)
    return filepath

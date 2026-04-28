from __future__ import annotations

import numpy as np
import pandas as pd

from archetype_workflow_utils import PATHS, append_log, ensure_dirs


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = (~np.isnan(values)) & (~np.isnan(weights)) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cum_weights = np.cumsum(weights)
    cutoff = quantile * weights.sum()
    idx = np.searchsorted(cum_weights, cutoff, side="left")
    idx = min(idx, len(values) - 1)
    return float(values[idx])


def safe_int(value, default: int = 0) -> int:
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(num):
        return default
    return int(num)


def safe_float(value, default: float = np.nan) -> float:
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(num):
        return default
    return float(num)


def comp_support_band(match_count: int, similarity_mean: float, q40: float, q70: float) -> str:
    if match_count >= 4 and pd.notna(similarity_mean) and similarity_mean <= q40:
        return "high"
    if match_count >= 3 and pd.notna(similarity_mean) and similarity_mean <= q70:
        return "medium"
    if match_count >= 2:
        return "low"
    if match_count == 1:
        return "very_low"
    return "insufficient"


def interpret_market_row(row: pd.Series) -> str:
    match_count = safe_int(row.get("comp_salary_match_count"), default=0)
    if match_count == 0:
        return (
            "Realistic comp neighborhood is available, but too few comparable Year-5 salary matches survive the merge to produce a credible comp-market anchor."
        )

    p25 = safe_float(row.get("comp_salary_anchor_p25"))
    p50 = safe_float(row.get("comp_salary_anchor_p50"))
    p75 = safe_float(row.get("comp_salary_anchor_p75"))
    support = row.get("comp_salary_anchor_support", "insufficient")
    macro = row.get("macro_archetype", "n/a")
    drift = row.get("identity_drift_class", "n/a")

    if pd.notna(p25) and pd.notna(p50) and pd.notna(p75):
        return (
            f"This player sits in the {macro} market neighborhood. "
            f"Matched realistic comps imply a Year-5 salary range around {p25:.3f} to {p75:.3f} of cap, "
            f"with a weighted midpoint near {p50:.3f}. "
            f"Comp-market support is {support}, and the early-career drift class is {drift}."
        )

    return (
        f"This player sits in the {macro} market neighborhood. "
        f"Comparable salary support is partial rather than complete, so the comp-market anchor should be treated cautiously. "
        f"Comp-market support is {support}, and the early-career drift class is {drift}."
    )


def build_comp_salary_detail(realistic: pd.DataFrame, year5: pd.DataFrame) -> pd.DataFrame:
    target_small = year5[
        [
            "PLAYER_ID",
            "PLAYER_NAME",
            "year5_salary_cap_pct",
            "year_salary_total",
            "salary_cap",
            "year5_salary_match_flag",
            "salary_match_type",
        ]
    ].rename(
        columns={
            "PLAYER_ID": "comp_PLAYER_ID",
            "PLAYER_NAME": "comp_PLAYER_NAME_target",
            "year5_salary_cap_pct": "comp_year5_salary_cap_pct",
            "year_salary_total": "comp_year5_salary",
            "salary_cap": "comp_year5_salary_cap",
            "year5_salary_match_flag": "comp_year5_salary_match_flag",
            "salary_match_type": "comp_salary_match_type",
        }
    )

    detail = realistic.merge(target_small, on="comp_PLAYER_ID", how="left")
    detail["comp_name_alignment_flag"] = (
        detail["comp_PLAYER_NAME"].fillna("").str.strip().str.lower()
        == detail["comp_PLAYER_NAME_target"].fillna("").str.strip().str.lower()
    ).astype(int)

    eps = 1e-6
    detail["similarity_weight_raw"] = 1.0 / (pd.to_numeric(detail["similarity_score"], errors="coerce") + eps)
    detail["similarity_weight"] = detail.groupby("PLAYER_ID")["similarity_weight_raw"].transform(
        lambda s: s / s.sum() if s.notna().sum() and float(s.sum()) > 0 else np.nan
    )
    return detail


def build_block2_context(final_profile: pd.DataFrame, comp_detail: pd.DataFrame, year5: pd.DataFrame) -> pd.DataFrame:
    global_q40 = float(final_profile["realistic_comp_similarity_mean"].quantile(0.40))
    global_q70 = float(final_profile["realistic_comp_similarity_mean"].quantile(0.70))

    rows = []
    for player_id, grp in comp_detail.groupby("PLAYER_ID", dropna=False):
        matched = grp[grp["comp_year5_salary_match_flag"] == 1].copy()
        matched_count = int(len(matched))

        if matched_count > 0:
            vals = matched["comp_year5_salary_cap_pct"].to_numpy(dtype=float)
            w = matched["similarity_weight_raw"].to_numpy(dtype=float)
            finite_mask = np.isfinite(w) & (w > 0)
            if finite_mask.sum() == 0:
                w = np.full(len(matched), 1.0 / len(matched), dtype=float)
            else:
                w = np.where(finite_mask, w, 0.0)
                w = w / w.sum()
            weighted_mean = float(np.average(vals, weights=w))
            p25 = weighted_quantile(vals, w, 0.25)
            p50 = weighted_quantile(vals, w, 0.50)
            p75 = weighted_quantile(vals, w, 0.75)
            matched_comp_list = ", ".join(matched.sort_values("comp_rank")["comp_PLAYER_NAME"].astype(str).tolist())
        else:
            weighted_mean = np.nan
            p25 = np.nan
            p50 = np.nan
            p75 = np.nan
            matched_comp_list = ""

        rows.append(
            {
                "PLAYER_ID": player_id,
                "comp_salary_match_count": matched_count,
                "comp_salary_match_rate": matched_count / max(len(grp), 1),
                "comp_salary_anchor_weighted_mean": weighted_mean,
                "comp_salary_anchor_p25": p25,
                "comp_salary_anchor_p50": p50,
                "comp_salary_anchor_p75": p75,
                "matched_comp_salary_list": matched_comp_list,
                "comp_salary_anchor_support": comp_support_band(
                    matched_count,
                    float(pd.to_numeric(grp["similarity_score"], errors="coerce").mean()) if len(grp) else np.nan,
                    global_q40,
                    global_q70,
                ),
            }
        )

    anchor = pd.DataFrame(rows)
    out = final_profile.merge(anchor, on="PLAYER_ID", how="left")
    out = out.merge(
        year5[
            ["PLAYER_ID", "year5_salary_cap_pct", "year5_salary_match_flag", "salary_match_type"]
        ].rename(
            columns={
                "year5_salary_cap_pct": "historical_year5_salary_cap_pct_observed",
                "year5_salary_match_flag": "historical_year5_salary_observed_flag",
                "salary_match_type": "historical_year5_salary_match_type",
            }
        ),
        on="PLAYER_ID",
        how="left",
    )

    out["comp_salary_match_count"] = pd.to_numeric(out["comp_salary_match_count"], errors="coerce").fillna(0).astype(int)
    out["comp_salary_match_rate"] = pd.to_numeric(out["comp_salary_match_rate"], errors="coerce")
    for col in [
        "comp_salary_anchor_weighted_mean",
        "comp_salary_anchor_p25",
        "comp_salary_anchor_p50",
        "comp_salary_anchor_p75",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["block2_market_context_interpretation"] = out.apply(interpret_market_row, axis=1)
    out["ceiling_comp_note"] = np.where(
        out["ceiling_comp_supported"].fillna(0).astype(int) == 1,
        "Ceiling comp is upside-only context and should not be used as the primary pricing anchor.",
        "No supported ceiling comp surfaced for this profile.",
    )

    keep_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "draft_year",
        "macro_archetype",
        "shot_style_subtype",
        "hybrid_archetype_label",
        "prototype_fit_ambiguity",
        "identity_drift_class",
        "realistic_comp_list",
        "realistic_comp_similarity_mean",
        "matched_comp_salary_list",
        "comp_salary_match_count",
        "comp_salary_match_rate",
        "comp_salary_anchor_weighted_mean",
        "comp_salary_anchor_p25",
        "comp_salary_anchor_p50",
        "comp_salary_anchor_p75",
        "comp_salary_anchor_support",
        "median_comp_group_points",
        "median_comp_group_minutes",
        "median_comp_group_rebounds",
        "median_comp_group_assists",
        "ceiling_comp_PLAYER_NAME",
        "ceiling_comp_supported",
        "ceiling_comp_note",
        "supporting_shot_style_explanation",
        "block2_market_context_interpretation",
        "historical_year5_salary_cap_pct_observed",
        "historical_year5_salary_observed_flag",
        "historical_year5_salary_match_type",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    return out[keep_cols].copy()


def main() -> None:
    ensure_dirs()
    final_profile = pd.read_csv(PATHS.archetype_output_dir / "final_player_archetype_profile_table.csv")
    realistic = pd.read_csv(PATHS.archetype_output_dir / "realistic_comps.csv")
    year5 = pd.read_csv(PATHS.archetype_output_dir / "SalaryBlock/year5_salary_target_table.csv")

    comp_detail = build_comp_salary_detail(realistic, year5)
    block2 = build_block2_context(final_profile, comp_detail, year5)

    comp_detail_path = PATHS.archetype_output_dir / "SalaryBlock/comp_salary_detail_table.csv"
    block2_path = PATHS.archetype_output_dir / "SalaryBlock/deliverable3_block2_archetype_comp_market_context.csv"

    comp_detail.to_csv(comp_detail_path, index=False)
    block2.to_csv(block2_path, index=False)

    append_log(
        phase="PHASE 13 — BUILD BLOCK 2 ARCHETYPE AND COMP-MARKET CONTEXT",
        completed=(
            "Translated Deliverable 2 into a salary-ready Block 2 by merging realistic comps to the cleaned Year-5 salary-cap target table, "
            "building similarity-weighted comp-market anchors, and exporting a stakeholder-facing archetype-plus-comp-market context table."
        ),
        learned=(
            "The archetype dossier becomes directly useful for salary support once realistic comps are expressed as a Year-5 salary-cap neighborhood rather than only as stylistic or later-performance analogs."
        ),
        assumptions=(
            "Realistic comps remain the primary pricing anchor, while ceiling comps remain upside-only context. "
            "Similarity weighting uses 1 / similarity_score and the comp-support band is based on both matched-comp count and overall comp quality."
        ),
        files_read=[
            str(PATHS.archetype_output_dir / "final_player_archetype_profile_table.csv"),
            str(PATHS.archetype_output_dir / "realistic_comps.csv"),
            str(PATHS.archetype_output_dir / "SalaryBlock/year5_salary_target_table.csv"),
        ],
        files_written=[
            str(comp_detail_path),
            str(block2_path),
        ],
    )


if __name__ == "__main__":
    main()

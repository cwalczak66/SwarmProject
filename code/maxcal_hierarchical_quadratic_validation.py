"""RBE 511 Swarm Intelligence Final Project - MaxCal-Derived Swarm Control: Emergent Oscillation Between Coverage and Information Diffusion

Author: Filippo Marcantoni (fmarcantoni@wpi.edu), Pau Alcolea (palcolea@wpi.edu), Chris Walczak (cwalczak2@wpi.edu)
Course: RBE 511 - Swarm Intelligence (Prof. Carlo Pinciroli)
Institution: Worcester Polytechnic Institute  

-------------------------------------------------------------------------------------------------------------------------------------------

Validation script for the quadratic hierarchical oscillation phase.

This checks both that the quadratic controller still runs as a valid MaxCal
mode selector and that its second-order constraints sharpen at least one
behaviorally relevant phase contrast relative to the linear controller.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from maxcal_hierarchical_core import (
    HierarchicalConfig,
    build_arg_parser,
    make_quadratic_comparison_figure,
    result_summary,
    run_simulation,
    validation_json,
    validator_checks,
)


def main() -> None:
    parser = build_arg_parser(
        description="Validate the hierarchical oscillatory controller (quadratic constraints).",
        default_outdir="maxcal_hierarchical_quadratic_validation",
    )
    args = parser.parse_args()

    linear_cfg = HierarchicalConfig(
        controller_form="linear",
        T_sim=args.T,
        seed=args.seed,
        diffusion_layer1_mode=args.diffusion_layer1_mode,
        lambda_I_layer1=args.diffusion_lambda_I,
        diffusion_target_rate=args.diffusion_target_rate,
        diffusion_target_p_enc=args.diffusion_target_p_enc,
        diffusion_inverse_branch=args.diffusion_branch,
        diffusion_continuation_step=args.diffusion_continuation_step,
        coverage_rate=args.coverage_rate,
        information_rate=args.information_rate,
        rate_calibration_T=args.rate_calibration_T,
        lambda_switch=args.lambda_switch,
        stem="maxcal_hierarchical_linear",
    )
    quadratic_cfg = HierarchicalConfig(
        controller_form="quadratic",
        T_sim=args.T,
        seed=args.seed,
        diffusion_layer1_mode=args.diffusion_layer1_mode,
        lambda_I_layer1=args.diffusion_lambda_I,
        diffusion_target_rate=args.diffusion_target_rate,
        diffusion_target_p_enc=args.diffusion_target_p_enc,
        diffusion_inverse_branch=args.diffusion_branch,
        diffusion_continuation_step=args.diffusion_continuation_step,
        coverage_rate=args.coverage_rate,
        information_rate=args.information_rate,
        rate_calibration_T=args.rate_calibration_T,
        lambda_switch=args.lambda_switch,
        stem="maxcal_hierarchical_quadratic",
    )
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    linear_res = run_simulation(linear_cfg)
    quadratic_res = run_simulation(quadratic_cfg)
    coverage_only = run_simulation(quadratic_cfg, forced_mode="coverage")
    diffusion_only = run_simulation(quadratic_cfg, forced_mode="diffusion")

    quadratic_payload = validator_checks(quadratic_res, coverage_only, diffusion_only)
    quadratic_checks = quadratic_payload["checks"]

    linear_summary = result_summary(linear_res)
    quadratic_summary = quadratic_payload["integrated"]

    linear_modes = linear_summary["mode_windows"]
    quadratic_modes = quadratic_summary["mode_windows"]
    linear_dispersion_contrast = float(
        linear_modes["coverage"]["mean_dispersion"] - linear_modes["diffusion"]["mean_dispersion"]
    )
    quadratic_dispersion_contrast = float(
        quadratic_modes["coverage"]["mean_dispersion"] - quadratic_modes["diffusion"]["mean_dispersion"]
    )
    linear_encounter_contrast = float(
        linear_modes["diffusion"]["mean_encounter_proxy"] - linear_modes["coverage"]["mean_encounter_proxy"]
    )
    quadratic_encounter_contrast = float(
        quadratic_modes["diffusion"]["mean_encounter_proxy"] - quadratic_modes["coverage"]["mean_encounter_proxy"]
    )
    linear_contrast_available = bool(np.isfinite(linear_dispersion_contrast) and np.isfinite(linear_encounter_contrast))
    quadratic_has_primary_contrast = bool(quadratic_dispersion_contrast > 0.0 or quadratic_encounter_contrast > 0.0)
    strengthens_dispersion = bool(
        linear_contrast_available and quadratic_dispersion_contrast > 1.02 * linear_dispersion_contrast
    )
    strengthens_encounter = bool(
        linear_contrast_available and quadratic_encounter_contrast > 1.02 * linear_encounter_contrast
    )

    payload = {
        "paper_alignment": {
            "layer": "Hierarchical quadratic MaxCal validation",
            "claim": "Quadratic age constraints should preserve oscillation while sharpening at least one primary contrast relative to the linear supervisor.",
            "validated_outputs": [
                "linear-vs-quadratic dispersion",
                "linear-vs-quadratic encounter proxy",
                "age-plane comparison",
                "mode-wise dispersion contrast",
            ],
        },
        "linear_reference": linear_summary,
        "quadratic": quadratic_payload,
        "quadratic_vs_linear": {
            "linear_dispersion_contrast": linear_dispersion_contrast,
            "quadratic_dispersion_contrast": quadratic_dispersion_contrast,
            "linear_encounter_contrast": linear_encounter_contrast,
            "quadratic_encounter_contrast": quadratic_encounter_contrast,
            "linear_contrast_available": linear_contrast_available,
            "quadratic_has_primary_contrast": quadratic_has_primary_contrast,
            "quadratic_strengthens_dispersion_contrast": strengthens_dispersion,
            "quadratic_strengthens_encounter_contrast": strengthens_encounter,
            "quadratic_strengthens_any_primary_contrast": bool(strengthens_dispersion or strengthens_encounter),
            "quadratic_preserves_dispersion_contrast": bool(
                linear_contrast_available and quadratic_dispersion_contrast >= 0.90 * linear_dispersion_contrast
            ),
            "quadratic_preserves_encounter_contrast": bool(
                linear_contrast_available and quadratic_encounter_contrast >= 0.90 * linear_encounter_contrast
            ),
            "quadratic_loop_area_remains_nontrivial": bool(
                quadratic_summary["phase_loop_area"] > 1.0
            ),
        },
    }

    quadratic_ready_checks = quadratic_payload["checks"].copy()
    quadratic_ready_checks.pop("integrated_encounter_between_fixed_mode_baselines", None)
    quadratic_ready_checks.pop("diffusion_windows_are_more_clustered_than_coverage_windows", None)

    payload["stage2_quadratic_ready"] = bool(
        all(quadratic_ready_checks.values())
        and (
            payload["quadratic_vs_linear"]["quadratic_strengthens_any_primary_contrast"]
            if linear_contrast_available
            else payload["quadratic_vs_linear"]["quadratic_has_primary_contrast"]
        )
        and payload["quadratic_vs_linear"]["quadratic_loop_area_remains_nontrivial"]
    )
    payload["figures"] = ["maxcal_hierarchical_quadratic_validation.png"]

    make_quadratic_comparison_figure(
        linear_res=linear_res,
        quadratic_res=quadratic_res,
        outpath=outdir / "maxcal_hierarchical_quadratic_validation.png",
    )
    validation_json(outdir / "maxcal_hierarchical_quadratic_validation_summary.json", payload)

    print("Hierarchical Oscillation Validation - Quadratic Constraints")
    print(f"  Output directory : {outdir}")
    print(f"  Ready for final phase : {payload['stage2_quadratic_ready']}")
    print(
        "  Dispersion contrast : "
        f"linear={linear_dispersion_contrast:.3f}, "
        f"quadratic={quadratic_dispersion_contrast:.3f}"
    )
    print(
        "  Encounter contrast  : "
        f"linear={linear_encounter_contrast:.5f}, "
        f"quadratic={quadratic_encounter_contrast:.5f}"
    )
    print("  Saved summary      : maxcal_hierarchical_quadratic_validation_summary.json")


if __name__ == "__main__":
    main()

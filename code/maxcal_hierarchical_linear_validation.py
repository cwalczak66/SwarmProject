"""RBE 511 Swarm Intelligence Final Project - MaxCal-Derived Swarm Control: Emergent Oscillation Between Coverage and Information Diffusion

Author: Filippo Marcantoni (fmarcantoni@wpi.edu), Pau Alcolea (palcolea@wpi.edu), Chris Walczak (cwalczak2@wpi.edu)
Course: RBE 511 - Swarm Intelligence (Prof. Carlo Pinciroli)
Institution: Worcester Polytechnic Institute  

-------------------------------------------------------------------------------------------------------------------------------------------

Validation script for the linear hierarchical baseline.

The linear controller should be numerically well formed before it is
used as a comparison object: mode probabilities must be valid, both Layer 1
fixed-mode baselines must run, and the integrated run should report the same
macroscopic observables used:

    A_C(t) = mean coverage age,  A_I(t) = mean Age of Information.

Oscillation diagnostics are still written, but they are not expected for this
baseline to pass because this linear controller is used to motivate the quadratic extension.
"""

from __future__ import annotations

from pathlib import Path

from maxcal_hierarchical_core import (
    HierarchicalConfig,
    build_arg_parser,
    make_validation_figure,
    run_simulation,
    validation_json,
    validator_checks,
)


def finite_metric(payload: dict, key: str) -> bool:
    value = float(payload[key])
    return value == value and value not in (float("inf"), float("-inf"))


def main() -> None:
    parser = build_arg_parser(
        description="Validate the hierarchical linear Layer 2 baseline.",
        default_outdir="maxcal_hierarchical_linear_validation",
    )
    args = parser.parse_args()

    cfg = HierarchicalConfig(
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
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    integrated = run_simulation(cfg)
    coverage_only = run_simulation(cfg, forced_mode="coverage")
    diffusion_only = run_simulation(cfg, forced_mode="diffusion")

    payload = validator_checks(integrated, coverage_only, diffusion_only)
    oscillation_checks = payload["checks"]
    linear_checks = {
        "mode_probabilities_are_valid": bool(oscillation_checks["mode_probabilities_are_valid"]),
        "integrated_run_has_arrivals": bool(payload["integrated"]["total_arrivals"] > 0),
        "coverage_only_run_has_arrivals": bool(payload["coverage_only"]["total_arrivals"] > 0),
        "diffusion_only_run_has_arrivals": bool(payload["diffusion_only"]["total_arrivals"] > 0),
        "fixed_mode_baselines_are_finite": bool(
            finite_metric(payload["coverage_only"], "mean_dispersion")
            and finite_metric(payload["diffusion_only"], "mean_dispersion")
            and finite_metric(payload["coverage_only"], "mean_encounter_proxy")
            and finite_metric(payload["diffusion_only"], "mean_encounter_proxy")
        ),
    }
    payload["linear_baseline_checks"] = linear_checks
    payload["linear_baseline_ready"] = bool(all(linear_checks.values()))
    payload["stage2_linear_ready"] = payload["linear_baseline_ready"]
    payload["oscillation_diagnostics"] = {
        "note": "These diagnostics are expected to motivate the quadratic controller; they are not required for the linear baseline to be valid.",
        "checks": oscillation_checks,
    }
    payload["paper_alignment"] = {
        "layer": "Hierarchical linear MaxCal validation",
        "claim": "Linear age constraints are the diagnostic baseline before the quadratic oscillatory controller; they should produce valid mode probabilities and comparable fixed-mode baselines, while oscillation metrics are reported as diagnostics.",
        "validated_outputs": [
            "integrated-vs-fixed-mode dispersion",
            "integrated-vs-fixed-mode encounter proxy",
            "age-plane trajectory",
            "mode-wise dispersion diagnostics",
        ],
    }
    payload["figures"] = ["maxcal_hierarchical_linear_validation.png"]

    make_validation_figure(
        integrated=integrated,
        coverage_only=coverage_only,
        diffusion_only=diffusion_only,
        outpath=outdir / "maxcal_hierarchical_linear_validation.png",
    )
    validation_json(outdir / "maxcal_hierarchical_linear_validation_summary.json", payload)

    print("Hierarchical Linear Baseline Validation")
    print(f"  Output directory : {outdir}")
    print(f"  Linear baseline ready : {payload['linear_baseline_ready']}")
    print(f"  Mode switches    : {payload['integrated']['n_switches']}")
    print(
        "  Mode fractions   : "
        f"coverage={payload['integrated']['mode_windows']['coverage']['window_fraction']:.3f}, "
        f"diffusion={payload['integrated']['mode_windows']['diffusion']['window_fraction']:.3f}"
    )
    print(
        "  Integrated means : "
        f"S={payload['integrated']['mean_dispersion']:.3f}, "
        f"E={payload['integrated']['mean_encounter_proxy']:.5f}, "
        f"A_c={payload['integrated']['mean_cov_age']:.3f}, "
        f"A_i={payload['integrated']['mean_info_age']:.3f}"
    )
    print("  Saved summary    : maxcal_hierarchical_linear_validation_summary.json")


if __name__ == "__main__":
    main()

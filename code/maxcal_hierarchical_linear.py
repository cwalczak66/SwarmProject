"""RBE 511 Swarm Intelligence Final Project - MaxCal-Derived Swarm Control: Emergent Oscillation Between Coverage and Information Diffusion

Author: Filippo Marcantoni (fmarcantoni@wpi.edu), Pau Alcolea (palcolea@wpi.edu), Chris Walczak (cwalczak2@wpi.edu)
Course: RBE 511 - Swarm Intelligence (Prof. Carlo Pinciroli)
Institution: Worcester Polytechnic Institute  

-------------------------------------------------------------------------------------------------------------------------------------------

Run the hierarchical MaxCal supervisor with linear Layer 2 constraints.

The Layer 2 mode selector samples

    Pr(m) proportional to exp[-L_m],
    L_m = lambda_C a_C^+(m) + lambda_I a_I^+(m) + lambda_switch 1[m != previous],

where ``a_C^+`` and ``a_I^+`` are the predicted next normalized coverage age
and Age of Information.  The run is kept as the diagnostic time-homogeneous
baseline before the quadratic controller.
"""

from __future__ import annotations

from pathlib import Path

from maxcal_hierarchical_core import (
    HierarchicalConfig,
    build_arg_parser,
    result_summary,
    run_simulation,
    save_result_bundle,
)


def main() -> None:
    parser = build_arg_parser(
        description="Run the hierarchical MaxCal controller with linear Layer 2 constraints.",
        default_outdir="maxcal_hierarchical_linear",
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
    result = run_simulation(cfg)
    summary = save_result_bundle(result, outdir)

    print("Hierarchical MaxCal Oscillation - Linear Constraints")
    print(f"  Output directory    : {outdir}")
    print(f"  Total arrivals      : {result.total_arrivals}")
    print(f"  Mode switches       : {summary['n_switches']}")
    print(f"  Mean dispersion     : {summary['mean_dispersion']:.3f}")
    print(f"  Mean encounter      : {summary['mean_encounter_proxy']:.5f}")
    print(f"  Mean coverage age   : {summary['mean_cov_age']:.3f}")
    print(f"  Mean information age: {summary['mean_info_age']:.3f}")
    print(f"  Dominant frequency  : {summary['dominant_frequency']:.4e}")
    print(f"  Spectral prominence : {summary['spectral_prominence']:.2f}")
    print(f"  Phase-loop area     : {summary['phase_loop_area']:.3f}")
    print("  Saved figures       :")
    print("    maxcal_hierarchical_linear_main.png")
    print("    maxcal_hierarchical_linear_age_plane.png")
    print("    maxcal_hierarchical_linear_psd.png")
    print("    maxcal_hierarchical_linear.gif")
    print("    maxcal_hierarchical_linear_summary.json")


if __name__ == "__main__":
    main()

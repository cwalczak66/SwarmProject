"""RBE 511 Swarm Intelligence Final Project - MaxCal-Derived Swarm Control: Emergent Oscillation Between Coverage and Information Diffusion

Author: Filippo Marcantoni (fmarcantoni@wpi.edu), Pau Alcolea (palcolea@wpi.edu), Chris Walczak (cwalczak2@wpi.edu)
Course: RBE 511 - Swarm Intelligence (Prof. Carlo Pinciroli)
Institution: Worcester Polytechnic Institute  

-------------------------------------------------------------------------------------------------------------------------------------------

Run the hierarchical MaxCal supervisor with quadratic Layer 2 constraints.

It keeps the same Layer 1 coverage and diffusion kernels as the linear baseline, 
but the Layer 2 mode cost becomes

    L_m = lambda_C a_C^+(m) + lambda_I a_I^+(m)
          + lambda_C2 [a_C^+(m)]^2 + lambda_I2 [a_I^+(m)]^2
          + lambda_switch 1[m != previous].

The squared age terms make mode probabilities depend on where the swarm is in
the coverage-age/AoI plane, which is the mechanism tested for oscillation.
"""

from __future__ import annotations

from pathlib import Path

from maxcal_hierarchical_core import (
    HierarchicalConfig,
    build_arg_parser,
    run_simulation,
    save_result_bundle,
)


def main() -> None:
    parser = build_arg_parser(
        description="Run the hierarchical oscillatory MaxCal controller (quadratic constraints).",
        default_outdir="maxcal_hierarchical_quadratic",
    )
    args = parser.parse_args()

    cfg = HierarchicalConfig(
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
    result = run_simulation(cfg)
    summary = save_result_bundle(result, outdir)

    print("Hierarchical MaxCal Oscillation - Quadratic Constraints")
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
    print("    maxcal_hierarchical_quadratic_main.png")
    print("    maxcal_hierarchical_quadratic_age_plane.png")
    print("    maxcal_hierarchical_quadratic_psd.png")
    print("    maxcal_hierarchical_quadratic.gif")
    print("    maxcal_hierarchical_quadratic_summary.json")


if __name__ == "__main__":
    main()

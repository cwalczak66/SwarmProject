# maxcal_coverage.jl
#
# MaxCal Coverage Simulation — Julia implementation
# Theory: Sec. 4.1–4.2.1 of "From Collective Specifications to Local Behaviors"
#
# SETUP (run once in Julia REPL before first use):
#   using Pkg
#   Pkg.add(["Plots"])          # Plots.jl + GR backend
#   Pkg.add(["CairoMakie"])     # alternative for publication figures
#
# RUN:
#   julia> include("maxcal_coverage.jl")
#   or:  julia maxcal_coverage.jl

using Random
using Statistics
using LinearAlgebra
using Plots
gr()   # GR backend: fast, interactive; swap for pyplot() or CairoMakie as needed

# ============================================================
# § 0  PARAMETERS  (single source of truth — change only here)
# ============================================================

const NX           = 20        # grid columns
const NY           = 20        # grid rows
const CELL_SIZE    = 1.0       # metres per cell → world is 20 × 20 m
const N_ROBOTS     = 50
const ROBOT_SPEED  = 0.15      # m / simulation step
const T_SIM        = 12_000    # total simulation steps
const RECORD_EVERY = 60        # steps between c_k(t) snapshots
const SNAP_EVERY   = 300       # steps between position snapshots
const LAMBDA_C_VAL = 0.0       # symmetric multiplier (see §1 note)
const SEED         = 42

# NOTE on LAMBDA_C_VAL = 0:
#   Theory Eq.(1): p*(k2|k1) = w_{k2,k1}·exp(−λ_C^{k2}) / Z(k1)
#   With equal multipliers ∀k, exp(−λ_C) cancels in the ratio,
#   giving a uniform random walk: p*(k2|k1) = 1 / |N(k1)|.
#   The stationary distribution is then π_k ∝ deg(k) (see §3).
#   We keep the full formula so non-uniform λ_C^k slots in later.


# ============================================================
# § 1  WORLD — discrete region graph
# ============================================================

struct World
    Nx       :: Int
    Ny       :: Int
    K        :: Int              # total regions = Nx * Ny
    cell_size:: Float64
    adjacency:: Vector{Vector{Int}}   # 1-indexed; 8-connected
end

function build_world(Nx, Ny, cell_size)
    K   = Nx * Ny
    adj = [Int[] for _ in 1:K]

    for k in 1:K
        row, col = divrem(k - 1, Nx)   # 0-based within this function
        for dr in -1:1, dc in -1:1
            (dr == 0 && dc == 0) && continue
            nr, nc = row + dr, col + dc
            (0 <= nr < Ny && 0 <= nc < Nx) || continue
            push!(adj[k], nr * Nx + nc + 1)   # store 1-indexed
        end
    end

    World(Nx, Ny, K, cell_size, adj)
end

# Continuous-space centre of region k (1-indexed)
function region_center(w::World, k::Int)
    row, col = divrem(k - 1, w.Nx)
    x = (col + 0.5) * w.cell_size
    y = (row + 0.5) * w.cell_size
    return x, y
end


# ============================================================
# § 2  MAXCAL TRANSITION PROBABILITIES
# ============================================================
#
# Theory, Eq.(1):
#   p*(k2 | k1) = w_{k2,k1} · exp(−λ_C^{k2}) / Z(k1)
#   Z(k1)       = Σ_{k2 ∈ N(k1)} w_{k2,k1} · exp(−λ_C^{k2})
#
# Here w = 1 for all edges (isotropic cost, absorbed into adjacency).
# We pre-compute the full K×K matrix so the inner loop is a table look-up.
# Later: replace or augment lambda_C to add λ_E (energy) or λ_I (info).

function build_transition_matrix(w::World, lambda_C::Vector{Float64})
    P = zeros(Float64, w.K, w.K)
    for k1 in 1:w.K
        neighbors = w.adjacency[k1]
        scores    = [exp(-lambda_C[k2]) for k2 in neighbors]
        Z         = sum(scores)
        for (i, k2) in enumerate(neighbors)
            P[k1, k2] = scores[i] / Z
        end
    end
    return P
end

# Sample from the categorical distribution over N(k1)
function sample_next_region(w::World, k::Int, P::Matrix{Float64},
                             rng::AbstractRNG)
    nb    = w.adjacency[k]
    probs = [P[k, k2] for k2 in nb]

    # Categorical sampler (avoids StatsBase dependency)
    r = rand(rng)
    s = 0.0
    for (i, p) in enumerate(probs)
        s += p
        s >= r && return nb[i]
    end
    return nb[end]   # fallback for floating-point edge case
end


# ============================================================
# § 3  THEORETICAL STATIONARY DISTRIBUTION
# ============================================================
#
# For a reversible MC with unit edge weights and equal λ_C^k,
# detailed balance π_k1 · P[k1,k2] = π_k2 · P[k2,k1] gives:
#
#   π_k1 / deg(k1) = π_k2 / deg(k2)  ⟹  π_k ∝ deg(k)
#
# This is the "ground truth" we compare the simulation against.
#
# Degree census for a 20×20 8-connected grid:
#   Corner cells (4 total):        degree 3
#   Non-corner edge cells (72):    degree 5
#   Interior cells (324):          degree 8

function theoretical_stationary(w::World)
    degrees = Float64[length(w.adjacency[k]) for k in 1:w.K]
    return degrees ./ sum(degrees)
end


# ============================================================
# § 4  ROBOT
# ============================================================
#
# Fields
#   from_k  — the Markov-chain state: last region the robot arrived at.
#             This is the variable s_t in the theory.
#   to_k    — target region (already sampled from p*(· | from_k)).
#   (tx,ty) — continuous-space centre of to_k.
#
# DESIGN CHOICE — "sample on arrival":
#   The Markov transition p*(k2|k1) fires ONLY when the robot
#   physically reaches the centre of to_k. Between transitions
#   the Markov state is frozen. Continuous motion is not part
#   of the Markov chain; it is only its physical realisation.
#
# DESIGN CHOICE — capped movement:
#   At each step the robot moves min(speed, dist) toward the target.
#   This prevents the oscillation that arises in discrete-time
#   steering when speed > 2 · tolerance: if we move a fixed step
#   and overshoot, the robot bounces forever just outside the
#   tolerance radius. Capping guarantees arrival in ⌈dist/speed⌉
#   steps exactly, making transit time a deterministic function
#   of geometry — as the theory implicitly assumes.

mutable struct Robot
    id     :: Int
    x      :: Float64
    y      :: Float64
    from_k :: Int     # Markov state
    to_k   :: Int     # next target region
    tx     :: Float64 # target centre
    ty     :: Float64
end

function make_robot(id::Int, w::World, P::Matrix{Float64}, rng)
    k0        = rand(rng, 1:w.K)
    cx, cy    = region_center(w, k0)
    k1        = sample_next_region(w, k0, P, rng)
    tx, ty    = region_center(w, k1)
    Robot(id, cx, cy, k0, k1, tx, ty)
end

function step!(r::Robot, w::World, P::Matrix{Float64},
               speed::Float64, markov_visits::Vector{Int}, rng)

    dx   = r.tx - r.x
    dy   = r.ty - r.y
    dist = sqrt(dx^2 + dy^2)

    # ── IN TRANSIT: capped movement ─────────────────────────────
    # Move toward target; never overshoot.
    move = min(speed, dist)
    if dist > 1e-10
        r.x += move * dx / dist
        r.y += move * dy / dist
    end

    # ── ARRIVAL TEST ─────────────────────────────────────────────
    # Position matches target to floating-point precision.
    if abs(r.x - r.tx) < 1e-9 && abs(r.y - r.ty) < 1e-9

        # Update Markov state
        r.from_k = r.to_k

        # Record arrival — this is the "visit" counted in c_k(t):
        #   c_k(t) = (# arrivals at k up to Markov-time t) / (total arrivals)
        markov_visits[r.from_k] += 1

        # ── MAXCAL TRANSITION: sample p*(k2 | from_k), Eq.(1) ───
        next_k = sample_next_region(w, r.from_k, P, rng)
        r.to_k = next_k
        r.tx, r.ty = region_center(w, next_k)
    end
end


# ============================================================
# § 5  MAIN SIMULATION
# ============================================================

struct SimResult
    w                   :: World
    pi_empirical        :: Vector{Float64}
    ck_history          :: Vector{Vector{Float64}}  # c_k(t) snapshots
    markov_step_history :: Vector{Int}              # total arrivals at each snapshot
    pos_snapshots       :: Vector{Tuple{Vector{Float64}, Vector{Float64}, Int}}
end

function run_simulation(; speed      = ROBOT_SPEED,
                          T          = T_SIM,
                          lambda_C_val = LAMBDA_C_VAL,
                          seed       = SEED)
    rng  = MersenneTwister(seed)
    w    = build_world(NX, NY, CELL_SIZE)
    lambda_C = fill(lambda_C_val, w.K)
    P        = build_transition_matrix(w, lambda_C)

    robots         = [make_robot(i, w, P, rng) for i in 1:N_ROBOTS]
    markov_visits  = zeros(Int, w.K)
    ck_history     = Vector{Vector{Float64}}()
    ms_history     = Int[]
    pos_snapshots  = Tuple{Vector{Float64}, Vector{Float64}, Int}[]

    for t in 1:T
        for r in robots
            step!(r, w, P, speed, markov_visits, rng)
        end

        total = sum(markov_visits)
        if t % RECORD_EVERY == 0 && total > 0
            push!(ck_history, markov_visits ./ total)
            push!(ms_history, total)
        end

        if t % SNAP_EVERY == 0
            push!(pos_snapshots,
                  ([r.x for r in robots], [r.y for r in robots], t))
        end
    end

    pi_emp = markov_visits ./ sum(markov_visits)
    SimResult(w, pi_emp, ck_history, ms_history, pos_snapshots)
end


# ============================================================
# § 6  VISUALISATION
# ============================================================

function make_main_figure(res::SimResult)
    w = res.w
    pi_theory = theoretical_stationary(w)
    pi_emp    = res.pi_empirical

    # Representative regions (one of each degree class)
    k_corner   = 1                           # (0,0) → degree 3
    k_edge     = NX ÷ 2 + 1                 # bottom-edge middle → degree 5
    k_interior = (NY ÷ 2) * NX + NX ÷ 2 + 1  # centre → degree 8

    clims = (minimum(pi_theory), maximum(pi_theory))

    p1 = heatmap(reshape(pi_theory, w.Nx, w.Ny)',
                 title="(a) Theoretical π̄_k ∝ deg(k)",
                 xlabel="col", ylabel="row",
                 color=:viridis, clims=clims, aspect_ratio=:equal)

    p2 = heatmap(reshape(pi_emp, w.Nx, w.Ny)',
                 title="(b) Empirical π̂_k",
                 xlabel="col", ylabel="row",
                 color=:viridis, clims=clims, aspect_ratio=:equal)

    err = abs.(pi_emp .- pi_theory)
    p3  = heatmap(reshape(err, w.Nx, w.Ny)',
                  title="(c) |π̂_k − π̄_k|",
                  xlabel="col", ylabel="row",
                  color=:Reds, aspect_ratio=:equal)

    # Coverage convergence: theory's Eq.(3) predicts rate ∝ 1/(t+1)
    ms = res.markov_step_history
    p4 = plot(title="(d) Coverage convergence\n(dashes = theory)",
              xlabel="Markov steps (arrivals)",
              ylabel="c_k(t)",
              legend=:right)

    for (k, label, col) in [
            (k_corner,   "Corner   (deg 3)", :red),
            (k_edge,     "Edge     (deg 5)", :darkorange),
            (k_interior, "Interior (deg 8)", :seagreen)]
        ck = [res.ck_history[i][k] for i in eachindex(res.ck_history)]
        plot!(p4, ms, ck, color=col, lw=2, label=label)
        hline!(p4, [pi_theory[k]], color=col, ls=:dash, lw=1.2, label="")
    end

    xs, ys, t_end = res.pos_snapshots[end]
    p5 = scatter(xs, ys,
                 xlims=(0, w.Nx * w.cell_size),
                 ylims=(0, w.Ny * w.cell_size),
                 markersize=3, alpha=0.7, color=:dodgerblue,
                 aspect_ratio=:equal,
                 title="(e) Robot positions  t=$t_end",
                 xlabel="x (m)", ylabel="y (m)", label="")

    plot(p1, p2, p3, p4, p5,
         layout = @layout([a b c; d{0.5w} e]),
         size   = (1400, 900),
         suptitle = "MaxCal Coverage (Sec. 4.2.1) — symmetric λ_C, 8-connected grid")
end

function make_phase_figure(; T=T_SIM)
    # Phase diagram: convergence speed vs robot speed.
    #
    # KEY INSIGHT: the theory's convergence rate 1/(t+1) (Eq.3) is
    # in MARKOV time (number of arrivals).  In SIMULATION time the
    # rate scales with robot speed: faster robots = more Markov steps
    # per simulation step = faster apparent convergence.
    # Left plot: Markov time → curves should be speed-independent.
    # Right plot: simulation time → curves spread out by speed.

    w         = build_world(NX, NY, CELL_SIZE)
    pi_theory = theoretical_stationary(w)
    k_int     = (NY ÷ 2) * NX + NX ÷ 2 + 1  # interior region
    speeds    = [0.05, 0.10, 0.20, 0.50, 1.00]

    p_m = plot(title  = "Convergence in Markov time\n(speed-independent prediction)",
               xlabel = "Total Markov steps",
               ylabel = "|c_k(t) − π̄_k|",
               yscale = :log10, legend = :topright)

    p_s = plot(title  = "Convergence in simulation time\n(faster robot → faster per step)",
               xlabel = "Simulation steps",
               ylabel = "|c_k(t) − π̄_k|",
               yscale = :log10, legend = :topright)

    for spd in speeds
        res = run_simulation(speed=spd, T=T)
        sim_ts = (1:length(res.ck_history)) .* RECORD_EVERY
        errs   = [max(abs(res.ck_history[i][k_int] - pi_theory[k_int]), 1e-8)
                  for i in eachindex(res.ck_history)]

        plot!(p_m, res.markov_step_history, errs, lw=2, label="speed=$spd")
        plot!(p_s, sim_ts, errs, lw=2, label="speed=$spd")
    end

    plot(p_m, p_s, layout=(1,2), size=(1200,520))
end

function make_animation(res::SimResult; fps=12)
    w         = res.w
    pi_theory = theoretical_stationary(w)
    bg        = reshape(pi_theory, w.Nx, w.Ny)'

    anim = @animate for (xs, ys, t) in res.pos_snapshots
        heatmap(bg, color=:YlOrRd, alpha=0.5,
                xlims=(1, w.Nx), ylims=(1, w.Ny),
                aspect_ratio=:equal, title="t = $t",
                xlabel="col", ylabel="row", colorbar=false)
        scatter!(xs ./ w.cell_size, ys ./ w.cell_size,
                 markersize=3, alpha=0.8, color=:navy, label="")
    end

    gif(anim, "maxcal_coverage.gif"; fps=fps)
    println("Saved maxcal_coverage.gif")
end


# ============================================================
# § 7  ENTRY POINT
# ============================================================

println("MaxCal Coverage Simulation")
println("  World   : $(NX)×$(NY) grid, cell=$(CELL_SIZE) m")
println("  Swarm   : $(N_ROBOTS) robots, speed=$(ROBOT_SPEED) m/step")
println("  Duration: $(T_SIM) steps")
println()

println("Running simulation...")
result = run_simulation()
println("  Done. Total Markov steps: $(result.markov_step_history[end])")
println("  Expected per robot: ~$(round(Int, result.markov_step_history[end] / N_ROBOTS))")
println()

println("Saving main figure → maxcal_coverage_main.png")
fig_main = make_main_figure(result)
savefig(fig_main, "maxcal_coverage_main.png")

println("Saving animation → maxcal_coverage.gif")
make_animation(result)

println("Saving phase diagram (runs $(5) additional simulations)...")
fig_phase = make_phase_figure()
savefig(fig_phase, "maxcal_coverage_phase.png")
println("  Saved maxcal_coverage_phase.png")

println()
println("Done. Open the .png files to inspect results.")
println("Expected: empirical π̂_k closely tracks theoretical π̄_k ∝ deg(k).")
println("          Interior cells visited ~8/5 ≈ 1.6× more than edge cells.")
println("          Corner cells visited ~3/8 ≈ 0.375× as often as interior.")

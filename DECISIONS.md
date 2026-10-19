# Decision log

## Made

| Date       | Decision | Rationale |
|------------|----------|-----------|
| 2026-10-11 | GitHub repository `vign-xsh/financial-contagion-sim`, private, default branch `main`. | Pre-authorized in the Phase 0 brief. |
| 2026-10-11 | Recreated the repository from scratch. The previous repository of the same name (Phase 0 + Phase 1 history) was deleted by the owner and the local folder emptied first. | Owner's instruction. |
| 2026-10-11 | Python `>=3.11`; development interpreter is CPython 3.14. | Brief: newest 3.x available locally. |
| 2026-10-11 | Environment is `.venv` created with `python -m venv`; pip + `pyproject.toml` (PEP 621, setuptools backend). | Pre-authorized. |
| 2026-10-11 | `src` layout, package `financial_contagion_sim`. | Pre-authorized. |
| 2026-10-11 | Runtime dependencies: networkx, numpy, scipy, matplotlib, plotly. Dev extra: pytest, ruff, mypy, build. | Pre-authorized. |
| 2026-10-11 | `pyproject.toml` is the single source of truth; `requirements.txt` contains only `-e .[dev]`. | Pre-authorized. |
| 2026-10-11 | Version is defined once, in `financial_contagion_sim.__version__`, and read by setuptools (`dynamic = ["version"]`). | Avoids two copies drifting apart. |
| 2026-10-11 | The argparse stub lives in `src/financial_contagion_sim/cli.py`; root `main.py` is a wrapper around it. This adds one file to the layout in the brief. | `[project.scripts]` needs an importable entry point inside the installed package; a root-level `main.py` is not installable under a `src` layout. |
| 2026-10-11 | mypy runs in `strict` mode on `src`; ruff lint rules `E, F, W, I, B, UP, SIM`, line length 88. | Start strict while there is no code to retrofit. |
| 2026-10-11 | Conventional Commits; never amend, rebase or force-push. One branch per phase, pull request to `main`. | Pre-authorized. |
| 2026-10-14 | Spec §4 shows `src/*.py` with a bare `src/__init__.py`. Deliberate deviation: the package lives at `src/financial_contagion_sim/`. | A bare `src/__init__.py` is not an importable package. |
| 2026-10-14 | Extra files not in spec §4: `DECISIONS.md`, `tests/test_smoke.py`, `docs/spec/`, and `src/financial_contagion_sim/cli.py` (see the 2026-10-11 entry). | Decision log, Phase 0 smoke checks, a versioned copy of the spec, and the installable CLI entry point. |
| 2026-10-14 | Extra dev tools not in the spec: ruff, mypy, build. | Linting, formatting, static typing and packaging checks. |
| 2026-10-14 | Naming: the spec uses ρ for correlation (`ρ_ij`) and for infection density (`ρ_∞`, `ρ_0`). Code uses `corr` for correlations and `infected_fraction` / `initial_infected_fraction` for densities. | Avoids one symbol meaning two things in code. |
| 2026-10-14 | Supersedes the 2026-10-11 `requirements.txt` entry: it now lists the six spec §4 packages with lower bounds matching `pyproject.toml`, which stays authoritative; development installs use `pip install -e ".[dev]"`. A smoke test guards against drift. | Spec §4 alignment. |
| 2026-10-14 | Phase 1 A1: signed networks are undirected simple `networkx.Graph` objects; `DiGraph` and multigraphs are rejected (`TypeError`). | The triad term A_ij·A_jk·A_ki in spec §3.2 is symmetric. Approved in the Phase 1 decision gate. |
| 2026-10-14 | Phase 1 A2: self-loops are rejected (`ValueError`). | Approved. |
| 2026-10-14 | Phase 1 A3: duplicate edges are impossible in a simple `Graph`; multigraphs are rejected. There is no public edge-list input in Phase 1. | Approved. |
| 2026-10-14 | Phase 1 A4: isolated nodes and disconnected graphs are allowed, without a warning. | Approved. |
| 2026-10-14 | Phase 1 A5: every edge has `"sign"`, exactly a Python `int` equal to +1 or −1; `bool`, `float`, `str` and NumPy integers are rejected. Signs may be flipped in place. | Approved. Keeps NumPy scalars out when later phases flip signs. |
| 2026-10-14 | Phase 1 A6: nodes are exactly the Python ints 0..n−1. | Approved. Later phases use them as array indices. |
| 2026-10-14 | Phase 1 A7: edges carry only `"sign"`; correlation graphs also carry `"distance"` = d_ij as a float. | Approved. |
| 2026-10-14 | Phase 1 A8: non-edges are absent; the A_ij = 0 convention for non-edges belongs to the energy phase. | Approved. |
| 2026-10-14 | Phase 1 B1 (resolves the r(t) Open item): r0 = positive edges / m, where m is the total edge count. The simulator's friendly link density r(t) (spec §5) must use the same definition. | Approved; owner's choice over positive / C(n,2). |
| 2026-10-14 | Phase 1 B2: exactly `round(r0 * m)` edges are positive (Python `round`, round-half-even on the float product). The positive edges are a seeded permutation over edges sorted by (u, v) with u < v, so the signs depend only on the edge set. | Approved. See the Open item on float rounding. |
| 2026-10-14 | Phase 1 B3: generators return a `Graph` and record `g.graph["r0_requested"]` and `g.graph["r0_realized"]`, rather than returning a tuple. | Approved. |
| 2026-10-14 | Phase 1 B4: r0 = 0 (all negative) and r0 = 1 (all positive) are valid. | Approved. |
| 2026-10-14 | Phase 1 B5: a graph with m = 0 is valid; `r0_realized` is `None`. | Approved. |
| 2026-10-14 | Phase 1 C1: correlation pair i < j is an edge iff abs(corr_ij) ≥ tau; tau is required and lies in [0, 1]. The upper triangle `corr[i, j]` is used, which matters only within the symmetry tolerance. | Approved. The spec gives no edge-selection rule. |
| 2026-10-14 | Phase 1 C2: edge sign = sign(corr_ij); corr_ij = 0 is never an edge, even at tau = 0. d_ij = sqrt(2(1 − corr_ij)) (spec §5) is stored as `"distance"` and never used for selection. | Approved. d_ij is unsigned, so it cannot select strongly negative correlations. |
| 2026-10-14 | Phase 1 C4: correlation input must be square, 2-D, at least 1×1, finite, symmetric within atol 1e-8, with a unit diagonal within 1e-8, and entries in [−1, 1]. Values within 1e-8 outside the bounds are clamped before the sqrt; anything further out is rejected. | Approved. |
| 2026-10-14 | Phase 1 C5: correlation input is a `numpy.ndarray` of integer or floating dtype only; no pandas. | Approved. |
| 2026-10-14 | Phase 1 C6 (resolves the "dynamic" correlation Open item): one correlation matrix per call. | Approved. |
| 2026-10-14 | Phase 1 C7: no data downloads or bundled datasets; tests use hand-built matrices only. | Approved. |
| 2026-10-14 | Phase 1 C8: correlation graphs record `g.graph["r0_realized"]` (`None` with no edges) but no `r0_requested`. | Approved. |
| 2026-10-14 | Phase 1 D1: public API is `generate_barabasi_albert(n, m, r0, rng)`, `generate_erdos_renyi(n, p, r0, rng)`, `graph_from_correlation(corr, tau)` and `validate_signed_graph(g) -> None`. Every loader validates its own output. | Approved. |
| 2026-10-14 | Phase 1 D1b: the BA parameter keeps the name `m` (edges per new node), documented as distinct from the total edge count in the r0 definition. | Approved. |
| 2026-10-14 | Phase 1 D2: `rng` is a `numpy.random.Generator` or a non-negative `int`, normalized with `default_rng`; no global seeding. The networkx generator gets `int(rng.integers(0, 2**31 - 1))`, then the same `rng` draws the sign permutation. ER uses `networkx.gnp_random_graph`. | Approved. networkx 3.7 does accept Generators; the explicit int seed keeps its draws separate from the sign draws. |
| 2026-10-14 | Phase 1 D3: `TypeError` for wrong types, `ValueError` for out-of-range values; no custom exception hierarchy. | Approved. |
| 2026-10-14 | Phase 1 D4: n ≥ 1; BA 1 ≤ m < n (so BA needs n ≥ 2); ER p ∈ [0, 1]; r0 ∈ [0, 1]; `bool` is rejected where an int or real is expected; NaN is rejected. | Approved. |
| 2026-10-14 | Phase 1 E1: `graph_builder` holds no SAIS state, energy or simulation logic, and no node-state attribute. | Approved. |
| 2026-10-14 | `types-networkx` added to the `[dev]` extra, so mypy strict type-checks networkx calls. | Owner's choice over a `type: ignore` or a mypy override. Dev-only; no runtime dependency. |
| 2026-10-14 | Resolves the r0 rounding Open item: keep Python `round` on the float product `r0 * m` (round-half-even on the float). The same rule is used for the initial infected count (Phase 2 A7). | Owner's choice. |
| 2026-10-14 | Phase 2 A1: SAIS states are a `numpy.int8` array indexed by node; `State` IntEnum with SUSCEPTIBLE = 1, ALERT = 0, INFECTED = −1 (spec §3.1); dtype, length and values are validated. | Approved in the Phase 2 decision gate. |
| 2026-10-14 | Phase 2 A2: `SAISParams` frozen dataclass; β > 0, 0 ≤ β_a < β (β_a == β raises), κ ≥ 0, δ ≥ 0, all finite, `bool` rejected. Rates are in 1/time. | Approved; the β_a bound is spec §3.1. |
| 2026-10-14 | Phase 2 A3 (resolves the §3.1 "Recapitalized" Open item): exactly the four transitions in the §3.1 text. The diagram's "Recapitalized" arrow is the I → S transition; there is no A → S or I → A. | Owner confirmed. |
| 2026-10-14 | Phase 2 A4: N_i counts infected neighbours on positive edges only; negative edges contribute nothing. | Approved; spec §3.1 ("defaulted friendly (+1) counterparties"). |
| 2026-10-14 | Phase 2 A5: total rate R == 0 means the chain is absorbed: not an error; the clock jumps to t_max with reason `"absorbed"`. The test `R <= 0` is exact because rates are sums of products of finite non-negative numbers. | Approved. |
| 2026-10-14 | Phase 2 A6: `t_max` is required (finite, ≥ 0) and `max_events` optional. dt ~ Exp(R); if t + dt > t_max the clock stops at t_max and the event is not applied (exact by memorylessness). Reasons: `"t_max"`, `"absorbed"`, `"max_events"`. | Approved. |
| 2026-10-14 | Phase 2 A7: `make_initial_state(n, initial_infected_fraction, rng)` infects exactly `round(ρ0·n)` nodes chosen by `rng`; all others S, none A; reports requested and realized fractions. | Approved. Initial A = 0 remains an Open default because the spec is silent. |
| 2026-10-14 | Phase 2 A8: `rng` is a `Generator` or non-negative int seed, as in Phase 1; no global state. | Approved. |
| 2026-10-14 | Phase 2 A9: direct-method selection by an O(n) cumulative scan; N updated in O(deg v) per event and never recomputed globally inside a run. A Fenwick tree (O(log n) selection) is documented, not built. | Approved. |
| 2026-10-14 | Phase 2 A10–A11: separate pure functions `compute_rates`, `sample_waiting_time`, `select_event`; `gillespie_step` (one event) and `run_gillespie` (to t_max). `run_gillespie` returns a frozen `SAISResult` (final states, time, event count, reason) with an opt-in log of `Event(time, node, old_state, new_state)`; no fixed-step snapshots; S(t)/A(t)/I(t) reconstruction is left to Phase 3. | Approved. |
| 2026-10-14 | Phase 2 A12: `refresh_after_edge_flip(n_pos, states, i, j, new_sign)` updates N_i and N_j in O(1) after an edge-sign flip. No edge dynamics are implemented. | Approved; hook for Phase 3. |
| 2026-10-14 | Phase 2 A13: `compute_rates` takes an explicit N array; `infected_positive_neighbors(g, states)` is the full O(n + m) recomputation (also the test oracle); `run_gillespie` builds N once and maintains it. | Approved. |
| 2026-10-14 | Phase 2 B1–B3: E_Δ sums unordered triples i < j < k over C(n,3), with A_ij ∈ {−1, 0, +1} (0 for non-edges). "Balanced = −1" holds only for complete graphs, and \|E_Δ\| ≪ 1 on sparse graphs (documented). E_Δ = 0.0 for n < 3. | Approved; spec §3.2. |
| 2026-10-14 | Phase 2 B4: exact engine = `int64` dense trace(A³)/6, integer sum and one division; sparse engine = common-neighbour triangle sum, validated against the exact engine and a brute-force oracle (tests only). `int64` because A³ overflows `int8`. | Approved. |
| 2026-10-14 | Phase 2 C1: an S or A node and an I node joined by a +1 edge score +1, by a −1 edge −1; every other pair (including I–I) and every non-edge scores 0. | Approved reading of the spec §3.2 table. |
| 2026-10-14 | Phase 2 C2 (resolves the E_p aggregation Open item): E_p = (sum over edges of E_ij^p) / C(n,2); E_p = 0.0 for n < 2. | Owner approved. A fixed denominator like E_Δ's C(n,3): E_p ∈ [−1, 1] and the local difference is exact. |
| 2026-10-14 | Phase 2 C3: α ∈ [0, 1], finite; α = 1 gives E_Δ, α = 0 gives E_p; E ∈ [−1, 1] follows. `total_energy` uses the sparse E_Δ engine. | Approved. |
| 2026-10-14 | Phase 2 D1–D2 (resolves the §3.3 sign-convention Open item): flips apply to existing edges only. dE_Δ = +2·A_ij(old)/C(n,3)·Σ_k A_ik·A_jk, which equals the spec's −2·A_ij(t″)/C(n,3)·Σ A_ik(t′)A_jk(t′) with t″ = after the flip and t′ = before. | Derived independently and verified against brute-force recomputation. |
| 2026-10-14 | Phase 2 D3: dE_p = −2·c_ij·A_ij(old)/C(n,2) with c_ij = 1 iff exactly one endpoint is I (states fixed); dE = α·dE_Δ + (1 − α)·dE_p. | Approved. |
| 2026-10-14 | Phase 2 D4: acceptance uses the full energy E, so dE includes dE_p. | Owner approved. |
| 2026-10-14 | Phase 2 D5: `accept_flip(delta_energy, rng)`: dE < 0 accepts, dE == 0 accepts with probability 0.5, dE > 0 rejects; no temperature; not wired into any loop. | Approved; spec §3.3. |
| 2026-10-14 | Phase 2 D6: ties are tested exactly. Both numerators are integers and a float α converts exactly to `Fraction`, so `delta_total_energy(..., exact=True)` returns an exact `Fraction` whose `== 0` needs no tolerance at any n. A float result is also available. | Owner approved, over an rtol = 1e-12 test. |
| 2026-10-14 | Phase 2 D7: dE_Δ costs O(min(deg i, deg j)) (O(n) on dense graphs); dE_p is O(1); the total is not O(1) in general, despite the title of spec §3.3. No common-neighbour cache. | Approved; stated in the module docstring. |
| 2026-10-14 | Phase 2 E1–E4: pure functions plus frozen dataclasses (`SAISParams`, `InitialState`, `Event`, `StepResult`, `SAISResult`); ValueError / TypeError only; graphs are never mutated by energy or rate functions; `graph_builder.py` unchanged (its private validators are imported, not copied). | Approved. |
| 2026-10-14 | Phase 2 tests: statistical checks use fixed seeds and 5σ bounds: waiting-time mean and variance (200,000 draws, R = 2.5), event frequencies vs rate shares (60,000 draws), isolated recovery time with mean 1/δ (20,000 trials, δ = 2), and tie acceptance rate 0.5 (100,000 decisions). Float energy differences are compared with recomputation within 1e-12 absolute; exact differences are compared as `Fraction`s with no tolerance. | Non-flaky by construction. |
| 2026-10-19 | Phase 3 F5 (approved amendment to Phase 2, separate commit `531ca10`): new public helpers `structural_triad_sum`, `spreading_pair_sum`, `energy_from_sums`, `flip_energy_numerators`, `state_change_pair_delta` (`energy_calculator`) and `apply_sais_event` (`sais_engine`). Existing functions now delegate to them; their behaviour is unchanged (all Phase 2 tests pass unmodified). | Owner approved Option 1: the simulator reuses Phase 2 logic instead of duplicating it. |
| 2026-10-19 | Phase 3 A1–A3: frozen `SimulationConfig` validated in `__post_init__` (no file loading). Physical parameters have no defaults: `n`, `topology` with exactly one of `ba_m` / `er_p`, `sais_params` (Phase 2 `SAISParams`), `alpha`, `r0`, `initial_infected_fraction`, `edge_update_rate`, `t_max`, `seed`. Control defaults: `max_events=None`, `stop_on_sais_absorbed=False`, `record="grid"`, `num_points=501`, `verify_every=None`. | Approved. Range checks reuse the Phase 1–2 validators. |
| 2026-10-19 | Phase 3 A4: ρ0 is the initial infected fraction; `make_initial_state` infects `round(ρ0·n)` nodes from the `initial_state` stream, the rest S, none A; requested and realized values are reported. | Approved. Initial A = 0 remains Open. |
| 2026-10-19 | Phase 3 A5: γ (`edge_update_rate`) is finite and ≥ 0, in 1/time; γ = 0 is pure SAIS. | Approved. |
| 2026-10-19 | Phase 3 A6: results carry `schema_version` (1), the config echo, and the package, Python, NumPy and networkx versions. | Approved. |
| 2026-10-19 | Phase 3 B1: only the signs of existing edges change (+1 ↔ −1); edges are never created, deleted or rewired. Interpretation of spec §1's "topology … edge sign reconfiguration". | Approved. |
| 2026-10-19 | Phase 3 B2 (resolves the edge-flip rate/scheduling Open item): one joint CTMC. Every edge proposes a flip at rate γ, so R_net = γ·m; R = R_sais + R_net; dt ~ Exp(R); the channel is SAIS with probability R_sais / R. | Approved default; the spec gives no rate or schedule. Alternatives (global rate, alternating phases, time-scale separation) not implemented. |
| 2026-10-19 | Phase 3 B3–B4: the proposed edge is uniform over edges sorted by (u, v). Acceptance computes (dS3, dP) once with `flip_energy_numerators`, forms the exact `Fraction` dE of the **full** E (Phase 2 D4) with `energy_from_sums`, and calls `accept_flip`; tests check it equals `delta_total_energy(..., exact=True)`. | Approved. |
| 2026-10-19 | Phase 3 B5: a rejected proposal is an event: it takes time, counts toward `max_events`, and is counted (`n_proposed`, `n_accepted`, `n_rejected`, `n_zero_accepted`). | Approved. |
| 2026-10-19 | Phase 3 B6: after an accepted flip, S3, P, the positive-edge count and N_i are updated in O(1) (`refresh_after_edge_flip`); N_i is never recomputed globally. | Approved. |
| 2026-10-19 | Phase 3 B7: on sparse graphs most flips have dE = 0 exactly and are accepted with probability 0.5, so those edges random-walk towards r = 0.5. Documented and tested; a modelling consequence, not a bug. | Approved. |
| 2026-10-19 | Phase 3 B8: ties have probability zero; the event after t_max is not applied. | Approved. |
| 2026-10-19 | Phase 3 B9: the SAIS rate array is cached and rebuilt (O(n)) only after a SAIS jump or an accepted flip touching an infected node; other proposals cost O(min deg). `verify_every` checks the cache against recomputation. | Approved. |
| 2026-10-19 | Phase 3 C1: r(t) = positive edges / m (Phase 1 B1); NaN throughout when m = 0 (Phase 1 reports `r0_realized = None` for the same case). | Approved. |
| 2026-10-19 | Phase 3 C2: E(t) is kept as exact integers S3 and P (E_Δ = −S3/C(n,3), E_p = P/C(n,2)), updated by exact integer changes on every flip and SAIS jump; no float drift. | Approved. |
| 2026-10-19 | Phase 3 C3 (resolves the time-series recording Open item): `record="grid"` samples the exact right-continuous state at `num_points` evenly spaced times in [0, t_max]; `record="events"` records t = 0 and every event (C3b: including rejected proposals). Recording draws no random numbers, so both modes give the same trajectory. | Approved. |
| 2026-10-19 | Phase 3 C4: when the run ends at t_max or is absorbed, records extend to t_max; when it stops on `max_events` or `sais_absorbed`, records end at the stop time. | Approved. |
| 2026-10-19 | Phase 3 C5: `SimulationResult` fields are as approved: series `times`, `S`, `A`, `I`, `r`, `E`, `E_delta`, `E_p`, `infected_fraction`; counters; `termination_reason`, `final_time`; initial/final states; `edges` with initial/final signs as arrays; requested/realized r0 and ρ0; `config`, `seed_entropy`, `schema_version`, `versions`. | Approved, including the added `E_delta` / `E_p` components and sign arrays instead of a graph copy. |
| 2026-10-19 | Phase 3 D1–D2: `t_max` is required; `max_events` counts all events. Reasons: `"t_max"`, `"max_events"`, `"absorbed"` (R = 0 exactly), `"sais_absorbed"` (only with `stop_on_sais_absorbed=True`, default `False`, so edges keep evolving after the infection dies out). | Owner confirmed D2. |
| 2026-10-19 | Phase 3 E1–E3: `SeedSequence(seed).spawn(5)` gives streams [graph, initial_state, clock, sais_selection, edge_updates]; each component draws only from its own stream. A caller's `SeedSequence` is copied, never consumed. Nodes ascending; edges sorted by (u, v). Reproducible for the same config and seed on the same machine and package versions. | Approved; tests replay the documented streams. |
| 2026-10-19 | Phase 3 F1–F4: `run_simulation(config)` and `simulate(graph, initial_states, …, seed=…)`; both work on copies and never mutate caller data; no progress callback; ValueError / TypeError only, errors propagate unmasked. | Approved. |
| 2026-10-19 | Phase 3 F6: `scripts/benchmark_simulator.py` reports measured events per second for fixed sizes and seed; no assertions. | Approved. |
| 2026-10-19 | Phase 3 tests: statistical checks use fixed seeds and 5σ bounds: first-event transition frequencies (6,000 runs), channel probability R_net/R (6,000 runs), tie acceptance on a path (~40,000 proposals), the time-averaged r on [T/2, T] with T = 10,000 against 0.5 (sd from the two-state chain autocorrelation, stated in the test), and γ = 0 vs `run_gillespie` final infected counts (two-sample z ≤ 5, 400 runs each). | Non-flaky by construction. |

## Open

| Raised     | Question | Notes |
|------------|----------|-------|
| 2026-10-11 | Which license, if any? | None declared; the repository is private. Needed before any public release. |
| 2026-10-11 | Should CI (GitHub Actions) run the checks on pull requests? | Workflows need explicit approval; not part of Phase 0. |
| 2026-10-11 | `python` and `pip` are not on PATH on the development machine (only the `py` launcher). | Harmless inside an activated `.venv`; bootstrap with `py -3 -m venv .venv`. |
| 2026-10-14 | Edge-flip rate and scheduling in the co-evolutionary loop are unspecified (§5). | **Resolved 2026-10-19 (Phase 3 B2, see Made).** |
| 2026-10-14 | Is r(t) "friendly link density" positive edges / m or positive edges / C(n,2)? | **Resolved 2026-10-14: positive / m (Phase 1 B1, see Made).** |
| 2026-10-14 | Initial fraction of nodes in state A is unspecified. | Default adopted in Phase 2 (A7): A = 0 initially, S = 1 − ρ_0. Still open: the spec gives no initial A fraction. |
| 2026-10-14 | Meaning of "dynamic" stock return correlation matrices in §5. | **Resolved 2026-10-14: one matrix per call (Phase 1 C6, see Made).** Sequences and rolling windows remain possible future work, listed below. |
| 2026-10-14 | Aggregation and normalization of E_p are unspecified (§3.2). | **Resolved 2026-10-14 (Phase 2 C2, see Made).** |
| 2026-10-14 | The §3.1 diagram shows a "Recapitalized" arrow near β_a, but the text lists only four transitions. | **Resolved 2026-10-14 (Phase 2 A3, see Made).** |
| 2026-10-14 | Sign convention of the §3.3 local formula: A_ij(t'') vs A_ij(t'). | **Resolved 2026-10-14: t″ = after the flip (Phase 2 D2, see Made).** |
| 2026-10-14 | Steady-state definition, the "clustering" and "natural immunization" metrics, and which two of (α, r_0, ρ_0) are the 3D surface axes (§5). | No default proposed. |
| 2026-10-14 | Phase 1 C3: MST or k-nearest-neighbour filtering of correlation graphs. | Not in the spec; not implemented. Possible future option. |
| 2026-10-14 | Phase 1 C6: sequences of correlation matrices and rolling windows. | Out of scope for Phase 1; possible future work. |
| 2026-10-14 | B2 rounds the float product `r0 * m`, which differs by one from round-half-even of the exact decimal product near half-way points: 14 of 19,701 cases for two-decimal r0 in [0.01, 0.99] and m ≤ 200 (e.g. r0 = 0.07, m = 150 gives 11, not 10). Keep the float product, or round the exact value of `Fraction(str(r0)) * m`? | **Resolved 2026-10-14: keep the float product (see Made).** |
| 2026-10-14 | Phase 3 must define how the S(t), A(t), I(t) time series are recorded from the Gillespie event log (Phase 2 A11). | **Resolved 2026-10-19 (Phase 3 C3, see Made).** |
| 2026-10-14 | Faster SAIS event selection (Fenwick tree, O(log n)) and a common-neighbour cache for O(1)-amortized dE_Δ. | Documented, not built. Phase 3 benchmark: about 57k events/s at n ≤ 1000 falling to about 29k at n = 5000, where the O(n) rate rebuild after each SAIS jump starts to dominate. |
| 2026-10-19 | Phase 3 D3: energy-convergence or steady-state stopping. | Not in the spec; not implemented. Steady-state estimation belongs to Phase 4. |
| 2026-10-19 | Phase 3 B2 alternatives: a global proposal rate independent of m, alternating SAIS/edge phases, or explicit time-scale separation. | Not implemented; possible future options if the joint per-edge rate proves unsuitable. |

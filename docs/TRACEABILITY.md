# Traceability matrix

Spec item → implementation → covering tests → status. Spec sections refer to
[docs/spec/specification.pdf](spec/specification.pdf); decisions refer to
[DECISIONS.md](../DECISIONS.md). Test ids are pytest node ids under `tests/`
(parametrized cases are listed by function). Every id below was checked
against `pytest --collect-only -q` in Phase 5.

Status values: **covered** (implemented and tested as specified or as
decided), **partially covered**, **uncovered**, **deviates** (a documented,
approved difference from the PDF's wording).

| # | Spec item | Implementation | Tests | Status |
|---|---|---|---|---|
| 1 | §3.1 states S = 1, A = 0, I = −1 | `sais_engine.State`, `_check_states` | `test_sais.py::test_state_codes_match_spec`, `test_sais.py::test_rejects_invalid_states` | covered |
| 2 | §3.1 the four transitions and their rates | `sais_engine.compute_rates`, `apply_sais_event` | `test_sais.py::test_rates_hand_computed_path`, `test_sais.py::test_rates_hand_computed_star`, `test_sais.py::test_rate_channels_per_state`, `test_sais.py::test_apply_sais_event_rejects_disallowed_transitions` | covered |
| 3 | §3.1 N_i counts infected neighbours on positive edges only | `sais_engine.infected_positive_neighbors`, `refresh_after_edge_flip` | `test_sais.py::test_infected_positive_neighbors_ignores_negative_edges`, `test_invariants_system.py::test_all_negative_edges_transmit_nothing`, `test_sais.py::test_refresh_after_edge_flip_matches_recomputation` | covered |
| 4 | §3.1 0 ≤ β_a < β | `SAISParams.__post_init__` | `test_sais.py::test_params_accept_boundaries`, `test_sais.py::test_params_reject_invalid`, `test_reliability.py::test_sais_parameter_boundaries` | covered |
| 5 | §3.1 "transition probabilities over discrete Δt" | CTMC with rates, Gillespie direct method (`gillespie_step`, `run_gillespie`, `simulator._simulate`) | `test_sais.py::test_waiting_time_mean_and_variance`, `test_sais.py::test_event_frequencies_match_rate_shares`, `test_sais.py::test_isolated_recovery_time_has_mean_one_over_delta`, `test_sais.py::test_selection_probabilities_are_bounded_and_sum_to_one` | deviates (rates, not Δt probabilities; README "spec wording" 1) |
| 6 | §3.2 E_p pair table | `energy_calculator.spreading_pair_sum`, `spreading_energy` | `test_energy.py::test_spreading_pair_table`, `test_energy.py::test_pair_table_covers_every_combination`, `test_energy.py::test_non_edge_contributes_nothing` | covered |
| 7 | E_p aggregation (not in spec) | sum over edges / C(n,2); 0 for n < 2 | `test_energy.py::test_spreading_hand_computed_four_nodes`, `test_energy.py::test_spreading_single_node_is_zero`, `test_energy.py::test_spreading_bounded_on_random_graphs` | covered (Phase 2 C2) |
| 8 | §3.2 E_Δ and the triad table | `structural_triad_sum`, `structural_energy`, `exact_structural_energy` | `test_energy.py::test_triad_sign_table`, `test_energy.py::test_hand_computed_k4`, `test_energy.py::test_normalization_by_c_n_3`, `test_energy.py::test_engines_match_oracle_on_random_graphs`, `test_energy.py::test_fewer_than_three_nodes_gives_zero` | covered |
| 9 | §3.2 "perfectly balanced gives E_Δ = −1" | as row 8 | `test_energy.py::test_sparse_graphs_have_small_energy`, `test_invariants_system.py::test_complete_positive_graph_rejects_every_flip` | deviates (holds for complete graphs only; README "spec wording" 3) |
| 10 | §3.2 α ∈ [0, 1], E = αE_Δ + (1−α)E_p | `total_energy`, `energy_from_sums` | `test_energy.py::test_total_energy_endpoints_and_mix`, `test_energy.py::test_total_energy_rejects_bad_alpha`, `test_energy.py::test_total_energy_bounded`, `test_invariants_system.py::test_alpha_one_topology_ignores_node_states` | covered |
| 11 | §3.3 local flip difference and its sign convention | `delta_structural_energy`, `delta_total_energy`, `flip_energy_numerators` | `test_energy.py::test_local_deltas_match_exact_recomputation`, `test_energy.py::test_both_flip_directions_and_double_flip_restore`, `test_energy.py::test_flip_numerators_equal_change_in_sums`, `test_invariants_system.py::test_local_flip_change_on_scale_free_and_dense_graphs` | covered (t'' = after the flip, Phase 2 D2) |
| 12 | §3.3 "Fast O(1)" | common-neighbour sum, O(min(deg i, deg j)) | documented in `energy_calculator` docstring and README; no O(1) test exists or is claimed | deviates (not O(1); README "spec wording" 2) |
| 13 | §3.3 acceptance rule | `accept_flip`; exact ties via `Fraction` | `test_energy.py::test_negative_delta_always_accepted`, `test_energy.py::test_positive_delta_never_accepted`, `test_energy.py::test_zero_delta_accepted_half_the_time`, `test_energy.py::test_exact_tie_detected_at_large_n`, `test_simulator.py::test_acceptance_uses_full_energy`, `test_invariants_system.py::test_alpha_zero_without_infection_makes_every_flip_a_tie` | covered |
| 14 | §5.1 Barabási–Albert / Erdős–Rényi with positive-link ratio r0 | `generate_barabasi_albert`, `generate_erdos_renyi` | `test_graph_builder.py::test_ba_edge_count_is_m_times_n_minus_m`, `test_graph_builder.py::test_r0_is_exact_round_half_even`, `test_graph_builder.py::test_seeding`, `test_graph_builder.py::test_generated_graph_satisfies_contract` | covered |
| 15 | §5.1 correlation matrices, d_ij = sqrt(2(1 − ρ_ij)) | `graph_from_correlation` | `test_graph_builder.py::test_correlation_hand_built_example`, `test_graph_builder.py::test_correlation_distance_formula_special_values`, `test_graph_builder.py::test_correlation_threshold_is_inclusive_and_symmetric_in_sign`, `test_integration.py::test_correlation_matrix_to_saved_metrics_and_png` | covered (edge rule decided, Phase 1 C1–C2; CLI input not implemented, row 21) |
| 16 | §5.2 co-evolutionary loop in continuous time | `simulator.run_simulation`, `simulate` (joint CTMC, Phase 3 B2) | `test_simulator.py::test_invariants_with_full_verification`, `test_simulator.py::test_first_event_channel_probability`, `test_simulator.py::test_unbalanced_triangle_accepts_only_the_first_flip`, `test_simulator.py::test_sais_rates_follow_accepted_flips`, `test_invariants_system.py::test_maintained_integers_equal_recomputation_every_event` | covered |
| 17 | §5.2 time series S(t), A(t), I(t), r(t), E(G_t) | `SimulationResult` series, `_Recorder` | `test_simulator.py::test_grid_and_event_records_describe_the_same_trajectory`, `test_invariants_system.py::test_records_match_exact_recomputation_at_sampled_events`, `test_simulator.py::test_graph_without_edges_has_nan_r` | covered |
| 18 | §5.3 steady-state ρ∞ (definition not in spec) | `sweep.estimate_steady_state`, `aggregate` | `test_sweep.py::test_constant_series_is_converged_exactly`, `test_sweep.py::test_exponential_decay_on_both_sides_of_the_tolerance`, `test_sweep.py::test_drifting_ramp_is_not_converged`, `test_sweep.py::test_metrics_converge_independently`, `test_sweep.py::test_aggregation_counts_min_rule_std_and_se` | covered (Phase 4 B3) |
| 19 | §5.3 2D time-series charts | `visualizer.plot_timeseries`, `save_figure` | `test_visualizer.py::test_each_line_is_exactly_its_result_array`, `test_visualizer.py::test_layout_labels_and_caption`, `test_visualizer.py::test_save_png`, `test_integration.py::test_config_to_artifacts_to_cli_round_trip` | covered |
| 20 | §5.3 3D Plotly surfaces as .html over (α, r0, ρ0) | `visualizer.plot_surface`, `write_surface_html`; `sweep.run_sweep` | `test_visualizer.py::test_surface_data_matches_sweep_arrays`, `test_visualizer.py::test_surface_orientation_cell_by_cell`, `test_visualizer.py::test_html_is_self_contained`, `test_integration.py::test_reduced_sweep_saved_and_plotted_data_agree` | covered (axes decided, Phase 4 B1) |
| 21 | §5.1 correlation input from the CLI | — | — | uncovered (not implemented; Phase 4 C9 Open; library path covered by row 15) |
| 22 | §5.3 "natural immunization" and "clustering" | — | — | uncovered (no metric in the spec; Phase 4 A9 Open; no output claims either) |
| 23 | §4 `main.py` CLI entry point (argparse) | `cli.main`, root `main.py`, console script | `test_cli.py::test_run_end_to_end_matches_library`, `test_cli.py::test_sweep_end_to_end`, `test_cli.py::test_invalid_input_exits_2_naming_the_option`, `test_cli.py::test_root_main_py_runs_as_a_script`, `test_readme_examples.py::test_readme_example_runs` | covered |
| 24 | §4 file layout | package under `src/financial_contagion_sim/`, extra modules and tools | `test_smoke.py::test_module_imports` | deviates (approved, 2026-10-14 rows and Phase 4 E1) |

Rows 21 and 22 are the only `uncovered` items; both are logged as Open in
DECISIONS.md and are excluded from every claim in the README. Every
`deviates` row has an approved decision-log entry.

## Consequence checks (Phase 5 B2)

These follow directly from the approved rules and are asserted in
`tests/test_invariants_system.py`:

| Check | Test |
|---|---|
| β_a = 0 and γ = 0: A nodes never become I and never leave A | `test_no_attenuated_default_keeps_alert_nodes_alert` |
| κ = 0: no S → A event | `test_zero_kappa_means_no_hedging` |
| δ = 0: no I → S event; I never decreases | `test_zero_delta_means_infection_never_decreases` |
| All edges negative, γ = 0: no S → I, A → I or S → A; I only decays | `test_all_negative_edges_transmit_nothing` |
| γ = 0: no proposals, signs unchanged | `test_gamma_zero_never_touches_signs` |
| ρ0 = 0 and γ = 0: absorbed immediately | `test_no_infection_and_no_flips_is_absorbed_immediately` |
| Complete all-positive graph, α = 1: every proposal rejected | `test_complete_positive_graph_rejects_every_flip` |
| α = 1: E = E_Δ exactly; the sign trajectory does not depend on SAIS rates | `test_alpha_one_topology_ignores_node_states` |
| α = 0 and no infected node: every proposal is an exact tie | `test_alpha_zero_without_infection_makes_every_flip_a_tie` |

## Mutation (sensitivity) check (Phase 5 B4)

Run in a temporary copy of the repository (the working tree is never
modified), full suite with `-x` per perturbation. Every perturbation was
caught:

| Perturbation | First failing test |
|---|---|
| Flip the sign of dE_Δ | `test_energy.py::test_local_deltas_match_exact_recomputation` |
| C(n,3) instead of C(n,2) in E_p | `test_energy.py::test_local_deltas_match_exact_recomputation` |
| Count negative edges in N_i | `test_integration.py::test_correlation_matrix_to_saved_metrics_and_png` |
| β instead of β_a for A → I | `test_invariants_system.py::test_no_attenuated_default_keeps_alert_nodes_alert` |
| Accept dE = 0 with probability 1 | `test_energy.py::test_zero_delta_accepted_half_the_time` |
| Accept dE > 0 | `test_energy.py::test_positive_delta_never_accepted` |
| Drop the N_i refresh after an accepted flip | `test_cli.py::test_sweep_end_to_end` |
| r(t) over C(n,2) instead of m | `test_invariants_system.py::test_records_match_exact_recomputation_at_sampled_events` |
| Transpose the Plotly z orientation | `test_integration.py::test_reduced_sweep_saved_and_plotted_data_agree` |
| Steady-state window start off by one point | `test_sweep.py::test_exponential_decay_on_both_sides_of_the_tolerance` |
| Steady-state halves split off by one point | `test_sweep.py::test_exponential_decay_on_both_sides_of_the_tolerance` |
| Swap two child RNG streams | `test_simulator.py::test_streams_follow_documented_spawn_order` |

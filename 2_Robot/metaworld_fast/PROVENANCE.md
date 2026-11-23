# Provenance

The implementation was created for `CODEX_RCMLQ_METAWORLD_FAST_T128_S5_KO.md` in an initially empty experiment directory. The named manuscript, `Ideal_Type_Finder(3).pdf`, was not present in the inspected project. Its sections and equations therefore have not been independently checked against this code. The execution specification, including its explicit equations, is the available mathematical source. Existing code outside this directory used incompatible Gaussian-prior/Laplace or generic-quantile assumptions and was not silently reused as the requested posterior engine.

## Definitions retained from the manuscript as supplied in the specification

- Uniform prior on `[-1,1]^4`, squared-distance multinomial logit including the candidate norm offset, and exact likelihood posterior target.
- RC's projected-median Fisher score, PQ's posterior quantiles, the single QI hypothetical-mean transformation, independent joint posterior draws for PS, full-covariance GC, and the original SG stage schedule.
- The seven physical scales `[0.125, 0.2, 0.32, 0.5, 0.8, 1.25, 2.0]`, smallest-scale tie rule, deterministic principal-direction tie rule, and Gaussian radial cap of 32 around the current mean.
- Actual displayed-query integrated EIG divided by the same method's own seven-scale reference, and the trace of the full posterior covariance. These metrics are distinct from Fisher score and realized parameter squared error.
- Numerical target: adaptive marginal integration for actual coordinate-factorized histories, joint tempered resample-move SMC with 4096 particles and ten Metropolis sweeps per resampling, full-history target correction, and independent numerical reference checks.

These statements identify the intended mathematical provenance; they do not assert an independently verified reproduction of unavailable manuscript source code.

## User-specified changes

- All seven settings have dimension 4 and exactly 128 answers per algorithm/interaction. Five interaction IDs, `0,1,2,3,4`, are preassigned for every setting and method.
- Only RC, PQ, QI, PS, GC and SG are included. The shared `M=4, beta=4` setting is run once and reused in the two plotting sweeps.
- The beta sweep is at **d=4**. The specification identifies the manuscript's original beta sweep as **d=8**; these are not the same experiment.
- The beta used by the oracle and inference agrees in every setting. This is response sensitivity under a correctly specified model, not robustness to misspecification.
- SG retains its original stage length: every actual query, `t=0,...,127`, has stage 0 and scale 1. No compressed SG variant is introduced.

## Design choices in this implementation

- A 50-action-step reach-v3 realization clip encodes four q coordinates using three Cartesian channels over time: `sin(pi*u)` on x, y and z, and `sin(2*pi*u)` on x. The gripper is fixed. A common bounded nominal sequence and one global gain define open-loop actions.
- Five physical contexts are shared across methods/settings. Exact candidate bytes and the complete pinned physical context define cache equivalence. Simulation is an asynchronous realization audit of the saved statistical questions, and does not provide the oracle responses.
- Persistent CPU workers perform headless simulator replay; statistical resources are selected by measured profiling and recorded in the run manifest. Installed dependency versions, hardware observations and code/configuration hashes belong to each run's artifacts.
- Numerical checks select interaction 0, all methods, in the shared anchor, M=16 and beta=16 settings before inspecting method rankings. They compare the same observed q/y histories at answer counts 0/32/64/128. The original saved seven reference questions are re-evaluated, rather than regenerating adaptive questions from the reference posterior.
- Marginal references start at 513 nodes and may be refined once to 1025. Joint references use two independent 8192-particle replays, with one bounded 16384-particle refinement if unresolved. Reference Monte Carlo SE is reported separately; with only two reference replicates it is imprecise. The dense small-case checks use 81×81 and 161×161 Gauss-Legendre integration in d=2.
- Marginal moments and EIG use Simpson integration on an adaptive sinh grid with its Jacobian. Continuous quantiles invert the antiderivative of a positive PCHIP density interpolant. Remeshing rebuilds the coordinate density from its entire actual likelihood history. The continuous CDF interpolation is an implementation-specific numerical choice checked by resolution comparisons.
- Reporting shows all five raw interaction curves and their arithmetic mean without smoothing, bootstrap bands, automatic p-values or clipping ratios above one. Every mean has a recorded effective interaction count.

Source-history reuse is recognized only when a run's metadata supplies `reused_from` and preserved original seed/configuration information. No compatible pre-existing history was found during initial inspection. New computations, resumed computations and simulator replay are distinct operations.

The scheduler's frozen manifest and run metadata are authoritative for exact seeds, API versions, resource settings and execution status. A planned action or missing artifact is not an executed rollout or passed check.

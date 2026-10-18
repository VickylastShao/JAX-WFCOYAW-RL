# Case roles and evidence boundaries

The three cases have different purposes and were trained separately. Their sizes do not constitute a transfer-learning sequence or a matched scaling-law experiment.

## Three turbines: reference-task and implementation verification

The three-turbine task connects the implementation to a published LES–SAC reference. Its staged training record, numerical checks and learner timing clarify what is preserved from the reference setup and which execution costs change. The comparison between compiled JAX learner blocks and the locally configured TorchRL 0.3.1 benchmark excludes LES evolution. In particular, the reported 42.03-fold learner-throughput ratio must not be described as a full-system or end-to-end LES speedup. The cross-system comparison with a reported published training time also differs in hardware and evaluation coverage.

## Nine turbines: graph-policy learning and controller comparison

The nine-turbine case tests a graph policy across fixed directions. It completed million-interaction learning and reports positive modeled gross-energy gains on a precursor independent of training. Under the same reported evaluation protocol, the static LES lookup controller has a higher mean gain than the learned policy. The complete comparison and feedback-related ablations remain in the scientific record.

The directions are imposed at initialization by rotating the layout; these are not continuously varying wind-direction trajectories. Evaluation windows share one independent inflow source. The case therefore demonstrates directional response and beneficial learned actions without establishing superiority to lookup control or an identified benefit of real-time feedback.

## 81 turbines: primary execution method and farm-scale outcome

The main contribution separates persistent physical histories from active flow workspace and synchronized global learner work. Fixed-context tests isolate workspace reuse, while matched-work profiles isolate GPU-allocation costs. A completed million-interaction training run for an 81-turbine farm demonstrates practical use of the execution design.

The frozen final controller is evaluated on independently seeded B/C turbulent sources. Positive modeled gross-energy gains are supported by downstream recovery exceeding upstream losses. The two sources are independent of one another; the four windows within each source are dependent. The evidence covers one optimization seed and one nominal wind regime.

The separate actuator-force recomputation diagnostic addresses how an implementation changes the memory–time trade-off. It is not a retrospective replacement for the reported training implementation. Saved-record checks, source histories and numerical diagnostics support reproducibility at their stated levels; complete flow-field reconstruction across hardware and full training restart remain outside the demonstrated validation.

# SafeCross benchmark design

## Purpose

SafeCross studies one narrow safety question:

> Can a motion-forecasting system identify a developing conflict with a pedestrian or cyclist earlier than simple baselines, without making the simulated vehicle unnecessarily cautious?

The benchmark deliberately separates three claims:

1. **Forecasting:** how well does the model represent plausible future motion?
2. **Risk detection:** how early and reliably does it identify a possible vehicle–vulnerable-road-user conflict?
3. **Closed-loop utility:** does using that risk signal improve a simulated vehicle's behaviour without unacceptable loss of progress or comfort?

No offline score alone is treated as evidence of driving safety. No simulation result is treated as real-vehicle validation.

## Alignment with the intended audience

Woven by Toyota publicly describes automated-driving development as an active learning loop involving fleet data, issue and scenario discovery, scenario retrieval, expert-guided training, evaluation, and delivery. Its current careers material also emphasizes 3D perception, multimodal data, prediction, motion planning, scalable training/evaluation, failure-mode inspection, and safety-centered validation. SafeCross is designed to demonstrate a small, reproducible version of that loop rather than imitate a production autonomy stack.

Primary references:

- [Woven automated driving](https://www.woven.toyota/en/technology/automated-driving/)
- [Woven careers and current technical areas](https://www.woven.toyota/en/careers/)
- [Woven Vehicle Perception role](https://www.woven.toyota/en/careers/detail/263e0dfb-6bbb-409a-a5c5-8ff661030649)
- [Woven ML Platform role](https://www.woven.toyota/en/careers/detail/4be9f9b4-962f-4b09-80ea-8e37ca563795)

## Evaluation layers

### Layer A: deterministic synthetic unit scenarios

Small deterministic scenarios validate geometry, timing, risk logic, and visualisation before any learned model is introduced.

Required cases:

- pedestrian crosses from clear view;
- pedestrian emerges from behind a parked vehicle;
- pedestrian approaches the curb but does not cross;
- cyclist enters from a blind junction;
- several agents are present but only one conflicts with the ego path;
- observation noise or temporary missing observations;
- late, fast crossing;
- no-conflict negative controls.

These scenarios are not evidence of model quality. They are executable specifications for the metric implementation.

### Layer B: public motion data

Use Argoverse 2 Motion Forecasting as the primary development dataset. It provides 250,000 eleven-second scenarios at 10 Hz from six US cities, with HD maps and multiple dynamic-object types. It is released under CC BY-NC-SA 4.0.

- [Argoverse 2 official overview](https://www.argoverse.org/av2.html)
- [Official motion-forecasting task guide](https://argoverse.github.io/user-guide/tasks/motion_forecasting.html)

Use the Waymo Open Motion Dataset only as a separately licensed external-validation track. It provides 103,354 twenty-second segments, 10 Hz tracks, map data, and pedestrian/cyclist/vehicle labels. It is governed by Waymo's non-commercial dataset agreement and is not open source.

- [Waymo Motion Dataset](https://waymo.com/open/data/motion/)
- [Waymo dataset facts](https://waymo.com/open/about/)
- [Waymo licence](https://waymo.com/open/terms/)
- [Waymo FAQ and real-vehicle claim limitation](https://waymo.com/open/faq/)

Do not combine the two sources into a redistributed dataset. Keep acquisition scripts, derived tables, model artifacts, and licence notices source-specific.

### Layer C: closed-loop simulation

Use CARLA ScenarioRunner for controlled pedestrian/cyclist interactions and repeatable perturbations. Use nuPlan only if its closed-loop planning stack materially reduces implementation risk; its official benchmark includes collision, off-road, time-to-collision, progress, human-similarity, comfort, and scenario-specific pedestrian/cyclist metrics.

- [CARLA official site and ScenarioRunner](https://carla.org/)
- [CARLA ScenarioRunner repository](https://github.com/carla-simulator/scenario_runner)
- [nuPlan official benchmark](https://www.nuplan.org/)

CARLA scenarios must be generated from explicit seeds. Each run records simulator version, map, weather, traffic seed, agent seed, model checkpoint, and risk-policy configuration.

## Scenario mining and derived labels

Candidate vulnerable-road-user interactions are mined using only past and ground-truth future trajectories available in the research split. Derived labels are computed reproducibly rather than hand-waved:

- minimum ego–agent separation;
- time of minimum separation;
- time to projected path conflict;
- post-encroachment time where meaningful;
- conflict-point location;
- whether the agent enters the ego corridor;
- occlusion proxy from track-observation history or simulator ground truth;
- agent type;
- interaction geometry;
- speed and acceleration context;
- warning opportunity: time between first defensible warning and conflict;
- difficulty tags: late appearance, sharp acceleration, multiple agents, map complexity, missing observations.

Thresholds are fixed from the training/development split and documented before reading final-test results. Sensitivity analysis reports results over a reasonable threshold range.

## Leakage-safe splitting

### Public-data development split

- Preserve official train/validation/test boundaries.
- Within official training data, group derived samples by original scenario identifier.
- Never allow windows from the same source scenario into different splits.
- Do not tune on leaderboard test responses.
- Report pedestrian and cyclist results separately.

### Generalisation splits

At least two of the following must be held out as entire groups:

- city;
- intersection or map region;
- interaction geometry;
- occlusion severity;
- agent class;
- scenario source;
- synthetic perturbation combination.

The main generalisation result uses a split fixed before model selection. A random-window split may be reported only as a weak diagnostic and never as the headline result.

### CARLA split

- development towns and routes are distinct from final evaluation towns/routes;
- final combinations of weather, occlusion, pedestrian speed, crossing angle, and traffic density are unseen jointly during tuning;
- scene seeds are stored and immutable;
- negative controls are matched to conflict scenarios by route and environment where possible.

## Baselines

Every layer includes understandable baselines.

### Forecasting

1. stationary agent;
2. constant velocity;
3. constant acceleration;
4. lane/map-aware kinematic baseline where map context applies;
5. one established learned public baseline reproduced from its official implementation;
6. SafeCross multi-hypothesis model.

### Risk detection

1. fixed distance threshold;
2. constant-velocity time-to-collision threshold;
3. deterministic forecast-to-risk conversion;
4. learned prediction without uncertainty calibration;
5. calibrated SafeCross risk;
6. oracle using ground-truth future, clearly labelled as an upper-bound diagnostic.

### Closed loop

1. no added risk policy;
2. conservative fixed-distance braking;
3. TTC-only policy;
4. SafeCross without calibration;
5. full calibrated SafeCross;
6. ground-truth-future oracle, diagnostic only.

## Metrics

### Forecasting metrics

- minADE and minFDE;
- miss rate;
- negative log likelihood when the output is probabilistic;
- Brier-style or probability-weighted trajectory error;
- calibration error for trajectory-mode probabilities;
- critical-mode recall: whether any predicted mode captures the future that produces the conflict;
- pedestrian and cyclist subgroup results.

Forecast metrics are supporting evidence. Lower minADE does not automatically imply a safer planner.

### Risk metrics

- area under the precision–recall curve;
- precision, recall, and F1 at a threshold selected only on development data;
- recall at a fixed false-warning rate;
- false warnings per 100 negative scenarios;
- expected calibration error and Brier score;
- median warning lead time;
- late-warning and missed-conflict rate;
- risk–coverage curve when abstention is enabled.

### Closed-loop metrics

- collision and near-miss rate;
- minimum separation and minimum TTC;
- successful yield rate;
- unnecessary intervention rate;
- hard-braking events;
- route completion/progress;
- added travel time;
- acceleration and jerk;
- off-road and traffic-rule infractions;
- performance under held-out perturbation combinations.

nuPlan's official benchmark is the reference for grouping planning metrics into traffic-rule compliance, human-driving similarity, vehicle dynamics/comfort, goal achievement, and scenario-specific measures: [nuPlan](https://www.nuplan.org/).

## Statistical analysis

- Run at least three independently seeded training runs for learned models.
- Evaluate all policies on the same frozen scenario seeds.
- Report counts and denominators, not percentages alone.
- Use paired bootstrap confidence intervals over scenarios for metric differences.
- Use 10,000 bootstrap resamples for headline differences unless computation is prohibitive.
- Report 95% confidence intervals.
- For rare binary events, include exact or Wilson intervals for absolute rates.
- Correct for multiple comparisons when many variants are tested; identify one primary endpoint in advance.
- Report both absolute and relative changes.
- Retain failed and interrupted simulator runs in an execution ledger; exclude only by predeclared technical-failure rules.

Primary endpoint:

> Difference in near-miss-or-collision rate between calibrated SafeCross and the strongest non-oracle baseline on the frozen held-out CARLA scenario suite.

Co-primary constraint:

> Route completion and comfort must not deteriorate beyond the predeclared non-inferiority margin.

## Required ablations

- map context removed;
- interaction modelling removed;
- multi-hypothesis output replaced with one trajectory;
- uncertainty calibration removed;
- occlusion feature removed;
- vulnerable-road-user-specific training removed;
- early-risk policy replaced by fixed TTC;
- data scale reduction;
- each public data source evaluated separately;
- weather/lighting/occlusion perturbations individually and jointly.

## Success thresholds

These are project gates, not promised outcomes.

### Minimum credible release

- all deterministic metric tests pass;
- at least 1,000 held-out closed-loop episodes;
- at least three training seeds;
- at least two non-oracle baselines;
- statistically supported improvement on one safety metric;
- no material regression in route completion;
- complete failure-case gallery and run manifest.

### Strong public result

At least one of the following on the frozen test suite, with a confidence interval excluding zero:

- 15% or greater relative reduction in near-miss-or-collision rate;
- 10 percentage-point or greater increase in critical-event recall at matched false-warning rate;
- 20% or greater reduction in false warnings at matched critical-event recall;
- 0.5 seconds or greater improvement in median warning lead time without a material rise in unnecessary interventions.

And all constraints:

- route completion absolute change no worse than -5 percentage points;
- comfort/hard-braking metric no worse than the predeclared 5% relative margin;
- result holds directionally across all three seeds;
- no subgroup is hidden: pedestrian, cyclist, visibility, and negative-control results are published.

Thresholds must not be retrofitted after seeing test data.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Forecast score improves but safety does not | Make closed-loop outcome the primary endpoint. |
| Model learns city or map identity | Hold out complete cities/regions and audit nearest neighbours. |
| CARLA policy becomes safe by stopping | Constrain route completion, delay, and comfort. |
| Rare-event percentages are unstable | Use large denominators, paired scenarios, and confidence intervals. |
| Simulator artefacts dominate | Randomise nuisance factors, publish failure cases, avoid real-world claims. |
| Licence contamination | Keep source-specific pipelines and do not redistribute source records. |
| Waymo-derived artifacts are mislicensed | Apply Waymo attribution and non-commercial downstream terms; publish access scripts/IDs rather than raw data. |
| Cherry-picked demo | Select demo scenes by predeclared criteria and link the full evaluation manifest. |
| Reproduction depends on hidden settings | Record versions, seeds, configs, checkpoints, and exact commands. |

## Reproducibility checklist

- environment lockfile;
- dataset download instructions and licence acknowledgement;
- scenario-index hashes;
- preprocessing checksums;
- fixed split manifests;
- model and policy configs;
- random seeds;
- raw per-scenario results;
- evaluation script;
- confidence-interval script;
- CARLA version and maps;
- render script that consumes saved results rather than hand-entered numbers;
- model card and dataset card;
- known-failure list.


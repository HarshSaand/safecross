# SafeCross-Interactions dataset card

## Status

**Planned derived benchmark and simulation dataset. No public release exists until the generation pipeline, licence audit, and validation checks are complete.**

## Summary

SafeCross-Interactions is designed for uncertainty-aware forecasting and risk evaluation around pedestrians and cyclists. It combines:

1. reproducible derived annotations that point to licensed public motion scenarios; and
2. independently generated CARLA interaction episodes with controlled perturbations and closed-loop outcomes.

The release is intended for research, education, and portfolio demonstration. It is not a safety certification dataset and must not be used to claim real-vehicle performance.

## Intended tasks

- pedestrian and cyclist motion forecasting;
- multi-agent interaction prediction;
- early conflict-risk detection;
- uncertainty calibration;
- abstention/selective prediction;
- scenario retrieval and difficulty analysis;
- closed-loop simulated yielding and braking evaluation.

## Source tracks

### Track A: Argoverse 2 references and derived annotations

[Argoverse 2 Motion Forecasting](https://www.argoverse.org/av2.html) contains 250,000 eleven-second scenarios at 10 Hz from six cities and is licensed under CC BY-NC-SA 4.0.

The SafeCross public repository should contain:

- source scenario identifiers;
- extraction code;
- versioned derived-label definitions;
- split manifests;
- small illustrative excerpts only when permitted;
- attribution and share-alike/non-commercial notices.

It should not silently relicense Argoverse-derived content under MIT or Apache.

### Track B: Waymo Open Motion references and derived annotations

[Waymo Open Motion](https://waymo.com/open/data/motion/) contains 103,354 twenty-second segments with tracks and map data. Access requires agreement to the [Waymo Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms/). Waymo states that the dataset is not open source, restricts production/vehicle operation, and governs derivative models and generated datasets.

The public repository should therefore contain:

- download/access instructions;
- source segment/scenario IDs where permitted;
- code that registered users run locally;
- attribution required by Waymo;
- explicit non-commercial downstream notice for Waymo-derived artifacts;
- aggregate evaluation results.

Do not redistribute raw Waymo data. Do not describe Waymo-derived weights or annotations as unrestricted open-source assets. Seek licence review before publishing any derived rows rather than only code and identifiers.

Waymo also warns that its data mixes manually and autonomously driven operation and is not suitable for evaluating the real-life performance of a specific vehicle: [official FAQ](https://waymo.com/open/faq/).

### Track C: CARLA-generated episodes

[CARLA](https://carla.org/) and [ScenarioRunner](https://github.com/carla-simulator/scenario_runner) are used to generate independently controlled scenarios.

Each episode records:

- simulator and ScenarioRunner versions;
- map and route;
- scenario template;
- random seed;
- weather, lighting, and visibility;
- ego state and control;
- pedestrian/cyclist states;
- map elements and conflict geometry;
- model forecasts and probabilities;
- risk score and policy decision;
- collision, near miss, yielding, progress, and comfort outcomes;
- render and provenance metadata.

CARLA assets and rendered outputs require a separate licence/provenance audit before choosing the final dataset licence.

## Data schema

### Scenario table

| Field | Description |
|---|---|
| `scenario_id` | Stable SafeCross identifier. |
| `source` | `argoverse2`, `waymo_motion`, or `carla`. |
| `source_version` | Dataset/simulator version. |
| `source_scenario_id` | Original identifier; never treated as SafeCross-owned raw data. |
| `split` | Train, development, validation, frozen test, or external validation. |
| `city_or_map` | Coarse source geography or simulator map. |
| `agent_type` | Pedestrian, cyclist, or other contextual agent. |
| `interaction_type` | Crossing, approach, merge, occluded emergence, non-conflict control, etc. |
| `visibility_bin` | Visible, partially occluded, strongly occluded, or unknown. |
| `conflict_label` | Derived conflict status under the versioned definition. |
| `min_separation_m` | Ground-truth or simulated minimum separation. |
| `time_to_conflict_s` | Time to defined conflict point, when applicable. |
| `warning_opportunity_s` | Earliest defensible warning time before conflict. |
| `difficulty_tags` | Multi-valued tags. |
| `derivation_version` | Hash/version of annotation code and thresholds. |
| `licence_track` | Source-specific licence routing identifier. |

### Time-series table

- timestamp;
- observed ego pose, velocity, acceleration, and heading;
- observed agent pose, velocity, acceleration, and heading;
- observation-validity mask;
- map/lane/crosswalk context;
- model trajectory modes and probabilities;
- predicted conflict probability;
- calibrated risk;
- policy state and action;
- ground-truth future where licence/split permits;
- closed-loop outcome for CARLA.

## Split design

- Preserve every official dataset boundary.
- Keep all windows from one source scenario together.
- Group by map region or city for the primary generalisation test.
- Freeze at least one interaction-geometry split.
- Freeze one CARLA test set with unseen joint perturbation combinations.
- Keep final-test labels inaccessible to the training command.
- Publish manifest hashes before headline experiments where practical.

Recommended release manifests:

- `train_standard`;
- `dev_standard`;
- `test_unseen_region`;
- `test_unseen_interaction`;
- `test_occlusion`;
- `test_negative_controls`;
- `carla_test_unseen_combinations`;
- `waymo_external_validation`.

## Annotation process

1. Run deterministic geometry-based mining.
2. Store continuous geometry values before thresholded labels.
3. Apply versioned thresholds fixed on development data.
4. Manually audit a stratified sample from every difficulty and source group.
5. Double-review ambiguous samples without using test outcomes to change definitions.
6. Record disagreement and correction logs.
7. Publish label-version migrations.

Synthetic occlusion labels use simulator ground truth. Public-data occlusion labels are proxies unless direct visibility evidence exists; the distinction must remain explicit.

## Quality checks

- duplicate source IDs across splits: zero;
- trajectory timestamp monotonicity: 100%;
- physically impossible velocity/acceleration flags reviewed;
- map-coordinate transform round-trip tolerance tested;
- metric implementations checked on deterministic fixtures;
- class and subgroup counts published;
- manual-audit precision for mined conflict labels reported;
- all final-test configuration hashes fixed before headline evaluation.

## Known limitations

- Public motion tracks do not expose every cue a real driver or production vehicle would observe.
- A derived crossing or near-miss label is definition-dependent.
- Occlusion from track history is not equivalent to camera visibility.
- CARLA physics, pedestrian behaviour, and sensor rendering do not establish real-world validity.
- US-city data do not represent global road behaviour.
- Pedestrians and cyclists are heterogeneous groups; aggregate metrics can hide subgroup failures.
- The dataset cannot establish the performance or safety of Toyota, Woven, Waymo, or any real vehicle.

## Ethical and safety considerations

- Do not infer sensitive personal attributes.
- Do not attempt to re-identify road users.
- Keep public-data privacy transformations intact.
- Report subgroup counts and failures where supported, without inventing demographic labels.
- Treat conservative behaviour as a trade-off, not automatic proof of safety.
- Include negative controls to penalise unnecessary interventions.
- Never deploy dataset-trained artifacts on a real vehicle.

## Licensing and release structure

Use a multi-track release rather than one blanket licence:

```text
code/                     project code licence
manifests/argoverse2/     CC BY-NC-SA 4.0 routing and attribution
manifests/waymo/          Waymo non-commercial routing and attribution
derived/waymo/            private-by-default pending licence review
carla_generation/         code licence plus asset/render provenance
results/                  result tables with source and licence metadata
```

Every model card states which sources trained the model and the resulting downstream restrictions.

## Citation and attribution

The final release must cite each upstream dataset using its official requested citation and reproduce required licence notices. For Waymo-derived work, include Waymo's prescribed attribution from its [terms](https://waymo.com/open/terms/).


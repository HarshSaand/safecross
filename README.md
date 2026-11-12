# SafeCross

SafeCross is an uncertainty-aware prediction and safety-evaluation project for
pedestrian and cyclist interactions. It asks a practical question:

> Can an automated vehicle identify a developing conflict early enough to yield,
> without reacting unnecessarily to every uncertain road user?

The first milestone is deliberately reproducible and lightweight. It provides
deterministic synthetic interaction scenarios, transparent baselines, calibrated
risk metrics, and a side-by-side visual demonstration. Later milestones connect
the same evaluation contract to Argoverse 2 or Waymo Open Motion data and to
closed-loop CARLA scenarios.

## Evidence boundary

Current results are generated from controlled synthetic scenarios. They do not
establish real-vehicle safety or production readiness. Public claims must identify
the dataset, split, baselines, simulator, number of runs, and uncertainty interval.

## Planned milestones

1. Synthetic, deterministic safety benchmark and visual demo.
2. Leakage-safe scenario mining from public motion datasets.
3. Multi-modal trajectory baseline with calibrated conflict risk.
4. Closed-loop CARLA evaluation under occlusion and domain shift.
5. Public scenario explorer, dataset card, model card, and technical report.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/render_demo.py --output outputs/safecross_demo.mp4
```

## Public foundations

- [Argoverse 2 Motion Forecasting](https://www.argoverse.org/av2.html)
- [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/)
- [nuPlan planning benchmark](https://www.nuplan.org/)
- [CARLA simulator](https://carla.org/)

## License

Code authored in this repository is intended for release under Apache-2.0.
Third-party datasets and assets retain their original licences and are not
redistributed here.

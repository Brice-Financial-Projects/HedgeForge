# Project Status — HedgeForge (Monte Carlo Engine)

## Current Phase
🧱 **Core design + docs complete**  
🚧 **Model + simulation engine implementation beginning**

## What’s Done
- Repo structured into `app/`, `src/`, `config/`, `tests/`
- Quarto documentation drafted (EDA, pipeline, methodology, risk metrics)
- Config loader + environment-aware YAML settings complete
- Architecture + simulation plan defined in `docs/`

## Next Milestones
| Milestone | Target |
|-----------|---------|
| Implement stochastic process module (GBM, OU) | Jan 2026 |
| Add Monte Carlo simulation engine + batching | Feb 2026 |
| Portfolio optimization + constraints layer | Mar 2026 |
| CLI + FastAPI wrapper (optional) | Later phase |

## Engineering Notes
- Designed as a backend engine, not a notebook-only model
- Documentation and modularity prioritized to support future ML V2 expansion

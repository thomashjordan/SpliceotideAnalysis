# legacy/

Superseded code, kept so the provenance of earlier results stays traceable. None
of it is used by the current pipeline, and it is not maintained.

| File | What it was |
|------|-------------|
| `main.py` | Original stage-1 preprocessor: parsed each mzML and wrote a per-sample `*_experiment.pkl` with its target peptides. `peak_picking_pipeline.py` now does both steps itself (target expansion lives in `AnalysisResults.expand_and_initialize_targets`). |
| `enzymeanalysis.py`, `enzymeplots.py` | Earlier ΔΔG / alignment / heat-map analysis. The functions still in use were ported into `plotting_analysis_functions.py`. |
| `testing_ground.ipynb` | Development scratch notebook. |

These files import the top-level modules, so run them from the repository root
with the root on `PYTHONPATH`, e.g. `PYTHONPATH=. python legacy/main.py sheet.xlsx`.

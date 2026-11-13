# NCM

Final NCM code and numerical tables for main Figures 2–6.

```bash
pip install -r requirements.txt
python plot.py
python replay_final.py --verify-only
```

| File or folder | Purpose |
|---|---|
| `train.py` | Final-stage fitting, continued fitting, prediction and result export. |
| `replay_final.py` | Reconstructs and verifies the selected final parameters without climate inputs. |
| `config.py` | Prepared-input paths, output folder and named forecast initialization. |
| `model/` | EOF/Markov backbone, neural correction, final-stage recipe and result export. |
| `training_recipe/` | Starting parameters and selected updates for the final model. |
| `plot.py`, `plotting/` | Draw main Figures 2–6 in PNG, PDF and editable SVG. |
| `figure_data/` | Saved final plotting tables and the verification climatology. |

For an exact parameter reconstruction, use `python replay_final.py --output results_replayed`. For fitting the archived final-stage protocol to matching prepared predictors, use `python train.py --mode fit-final --reuse-linear path/to/linear_results --output results_fitted`. The supplied final model is warm-started and includes development-event fitting; a new optimization run and a replay of the selected parameters are different operations.

`python train.py --mode predict --reuse-linear path/to/linear_results --output results` evaluates the supplied final parameters. `--mode continue-fit` runs additional historical fitting. `config.py` or CLI arguments select prepared fields and forecast inputs; raw climate data are not bundled.

The named initialization is the month after the last complete input month. Figure 4 uses February–July, Figure 5 September–February, and Figure 6 September 2026–February 2027. Plotting works directly from `figure_data/` without model fitting. Figure 1 is a separately prepared conceptual schematic.

Parameter replay verifies every stored numeric member. The compressed archive metadata may vary between NumPy versions.

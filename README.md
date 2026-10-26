# NCM

| File | Purpose |
|---|---|
| `train.py` | Runs the model calculation, prediction, and result export. |
| `plot.py` | Generates main Figures 2–5 in PNG, PDF, and SVG formats. |
| `config.py` | Sets input locations, output directories, and forecast origin. |
| `model/linear.py` | Fits EOF/MEOF and seasonal Markov models and builds predictors. |
| `model/neural.py` | Defines the neural network, embeds fitted parameters, and provides training and prediction functions. |
| `model/evaluation.py` | Calculates forecast metrics, event errors, and rankings and writes result tables. |
| `model/__init__.py` | Defines the model package. |
| `plotting/data.py` | Reads and checks the result tables used for plotting. |
| `plotting/figures.py` | Defines the layouts and plotting routines for Figures 2–5. |
| `plotting/style.py` | Sets fonts, colors, panel labels, and figure export settings. |
| `plotting/__init__.py` | Exposes the table-loading and figure-rendering functions. |
| `requirements.txt` | Lists the required Python packages. |

`figure_data/` contains the five numerical result tables used by the plotting code.

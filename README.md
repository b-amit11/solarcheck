# SolarCheck

[![Tests](https://github.com/b-amit11/solarcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/b-amit11/solarcheck/actions/workflows/ci.yml)

A PyTorch experiment for flagging potentially defective solar cells in electroluminescence (EL) images. Includes an original small CNN trained from scratch, a pixel-based logistic regression baseline, reproducible data splits, validation-only checkpoint and threshold selection, and a local Streamlit inspection interface.

**Stack:** Python · PyTorch · scikit-learn · NumPy · Streamlit · pytest · Docker · GitHub Actions

## Result first

**The CNN did not beat the simpler baseline.** On 445 held-out images, CNN average precision was 0.691 versus 0.728 for logistic regression. At a threshold chosen for at least 90% validation recall, the CNN found 130 of 143 positive cases (90.9% recall) but flagged 176 of 302 negative cases (58.3% false-positive rate). This is an educational experiment, not a production quality-control system.

The intended problem is reducing manual inspection effort. This experiment has not demonstrated such a reduction: 306 of 445 test images were flagged for review, and 13 positive cases were missed. No cost savings or deployment readiness are claimed.

## Run the saved model

Use Python 3.12 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8506
```

Open http://localhost:8506. The supplied checkpoint and test example images support inference without training or an API key. Upload only EL images of individual solar cells; regular photographs are outside the intended input domain. The app cannot reliably detect out-of-domain inputs.

## Reproduce the experiment

```bash
python -m solarcheck.prepare
python -m solarcheck.train
python -m pytest -q
```

Preparation loads the ELPV dataset from the installed `elpv-dataset` package. Training writes the model, metrics, split manifest, predictions, and learning history. Re-running training overwrites those artifacts. The recorded experiment used seed 42, CPU, 12 epochs, batch size 32, Adam with learning rate 0.001, and class-weighted binary cross-entropy. Installed versions are recorded in `requirements-lock.txt`; the broader requirements file supports installation on other platforms. Exact floating-point reproducibility across versions and devices is not guaranteed.

## Data and split protocol

ELPV contains 2,624 grayscale 300×300 cell crops from 44 photovoltaic modules, annotated with defect likelihood and mono/polycrystalline type. This project keeps labels 0 and 1 and excludes 401 intermediate-label images. The resulting **2,223 images** are split into **1,333 train / 445 validation / 445 test**, stratified jointly by label and cell type.

SHA-256 checks found no exact pixel duplicates among retained images. Exact-duplicate removal occurs before splitting. This does not exclude near-duplicate or correlated cells. **Module IDs are unavailable through the loader; cells from the same module may cross splits.** These results do not establish generalization to unseen modules. The binary extreme-label task also differs from the original dataset task.

The CNN uses three convolution/ReLU/pooling blocks, adaptive pooling, a dense layer with dropout, and a binary output. Images are resized to 96×96 grayscale and scaled to [0,1]; training adds random horizontal and vertical flips. No pretrained parameters or external model API are used.

The checkpoint with the highest validation average precision is selected. The threshold is then selected as the highest validation threshold achieving at least 90% recall. Only afterward is the final test set scored. The baseline uses standardized 16×16 pixel intensities and regularized logistic regression; its threshold is selected independently using the same validation rule.

## Evidence and files

- `reports/source.json`: package version, source attribution, raw archive checksum, label counts.
- `reports/split_manifest.csv`: original image index, class, cell type, split, exact image hash.
- `reports/history.json`: training loss and validation average precision by epoch.
- `reports/metrics.json`: CNN/baseline metrics, threshold, confusion matrix, and cell-type slices.
- `reports/test_predictions.csv`: individual held-out predictions for error analysis.
- `artifacts/model.pt`: state dictionary and preprocessing/threshold metadata; loaded with `weights_only=True`.

Example images are selected by fixed source-index order from the test split, with two images per class. They are not selected to showcase correct predictions.

## Critical assessment

The project demonstrates a real PyTorch forward/backward training loop, optimizer updates, imbalance handling, checkpoint serialization, dataset management, and evaluation. It does not show a new architecture, state-of-the-art performance, or commercial robustness. The baseline outperforming the CNN is a useful negative result and should remain visible.

The main limitations are high false-positive rates, excluded ambiguous labels, potentially correlated train/test cells, one random seed, uncalibrated scores, and lack of field validation. Further development should use a new locked test set or nested evaluation after this test set has been inspected. Suggested experiments include module-grouped splits if source metadata can be obtained, validation-only transfer-learning comparisons, uncertainty analysis, and higher-resolution inputs. Do not claim any of those experiments have already been completed.

## License and attribution

See `THIRD_PARTY.md`. ELPV images are **CC BY-NC-SA 4.0**, not commercially unrestricted. Supplied example images preserve that license; trained artifacts are offered only for noncommercial educational use. Original project code is MIT licensed. Respect separate upstream package licenses. No affiliation with the dataset authors is claimed.

Docker and CI configurations are supplied. Local Python checks were run; these files alone are not evidence of a successful Docker build, remote CI run, or public deployment.

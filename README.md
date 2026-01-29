# LESO — Latent Entropy-guided Synthetic Oversampling

LESO is a generic, standalone oversampling tool for **binary imbalanced classification** problems.
It implements the *Latent Entropy-guided Synthetic Oversampling* method proposed in our manuscript, combining **latent-state modeling** with **entropy-guided sample generation** to create informative synthetic minority samples while preserving data realism.

The tool operates directly on **CSV datasets**, making it easy for researchers and practitioners to experiment with LESO on real-world tabular data.

---

## Key Features

- Supports **binary classification** (minority vs. majority)
- Works directly on **CSV files**
- Allows the user to:
  - specify the class (label) column
  - identify minority and majority labels
  - choose a target minority proportion (e.g., balance the dataset)
  - exclude identifier or text-like columns from oversampling
- Treats:
  - numeric variables as continuous
  - categorical variables via one-hot encoding and decoding
- Preserves the **original dataset schema** in the output
- Includes a **safety rule** that prevents oversampling when the minority class is too small (fewer than 3 samples)

---

## Method Overview

LESO generates synthetic minority samples by:
1. Inferring **latent states** in the feature space using a probabilistic mixture model.
2. Computing **entropy-based weights** that identify ambiguous or under-represented regions.
3. Allocating synthetic samples preferentially to high-entropy latent states.
4. Interpolating between minority samples using entropy-modulated mixing coefficients.

Original observations are never modified; synthetic samples are appended to the dataset.

---

## Requirements

- Python ≥ 3.9
- numpy
- pandas
- scikit-learn

Install dependencies with:

```
pip install numpy pandas scikit-learn
```

---

## Usage

Run the tool from the command line:

```
python leso_csv_tool.py --input path/to/dataset.csv
```

Optional arguments:

```
--output path/to/output.csv   # default: <input>_LESO.csv
--seed   11                   # random seed (default: 11)
```

The tool will interactively prompt you to:
1. Select the class (label) column
2. Enter the minority and majority labels
3. Specify the target minority percentage relative to the majority class
4. Optionally exclude ID-like or text columns from oversampling

---

## Output

- A new CSV file containing:
  - all original observations (unchanged)
  - additional synthetic minority samples (if generated)
- Excluded columns are preserved in the output:
  - original rows keep their values
  - synthetic rows contain `NaN` in excluded columns
- Column order matches the input dataset

---

## Limitations

- **Binary classification only**
- Oversampling is disabled when the minority class has fewer than three samples
- Multiclass extension is left for future work

---

## Reproducibility

LESO is deterministic given a fixed random seed.
The tool is intended to support reproducible experimentation and comparative evaluation of oversampling methods.

---

## License

MIT License

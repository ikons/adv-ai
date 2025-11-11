# Advanced AI / Advanced Machine Learning Tutorials

This repository contains simple, well-documented Python implementations
of the algorithms taught in the **Advanced Artificial Intelligence**
course at the **Dept. of Informatics & Telecommunications, University of Thessaly**.

Goal: help students bridge **theory → practice** by providing:

- Jupyter notebooks with step-by-step training & inference.
- Python scripts (`.py`) that show cleaner, reusable code.

## Structure

- `decision_trees/` – Decision Trees for classification (Titanic, Mushrooms)
- `regression/` – Regression (planned)
- `bayesian_learning/` – Naive Bayes & Bayesian learning (planned)
- `bayesian_networks/` – Bayesian Belief Networks (planned)
- `association_rules/` – Association rules & Apriori (planned)
- `knn/` – k-Nearest Neighbors (planned)
- `svm/` – Support Vector Machines (planned)
- `clustering/` – Clustering (k-means, BIRCH, DBSCAN, silhouette) (planned)
- `notebooks/` – Jupyter notebooks, one per topic
- `data/` – Local data files (not tracked by git)

Large data files are **not** committed to git. Only `data/README.md` and
`data/.gitignore` are tracked.

## Installation

```bash
git clone https://github.com/ikons/adv-ai.git
cd adv-ai

python3 -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

### Windows / PowerShell notes

```powershell
# From the repository root (Windows PowerShell)
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# If Activate.ps1 is blocked by execution policy, run:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Running the notebooks

From the root of the repo:

```bash
jupyter notebook
# or
jupyter lab
```

Then open `notebooks/` and start with:

- `01_decision_trees_titanic.ipynb`

Each notebook explains which dataset it needs and how to run the experiments.

## Developer extras (recommended)

- `ipykernel` is included in `requirements.txt` so collaborators will get it when they install dependencies.
- After activating your venv, register a notebook kernel so VS Code / Jupyter can select it:

```powershell
# inside activated .venv
python -m ipykernel install --user --name adv-ai-venv-3.11 --display-name "adv-ai (.venv 3.11)"
```

- In VS Code open a notebook, click the kernel selector (top-right) and choose the registered kernel (display name above).

## Optional: pin the workspace interpreter

Create (or update) `.vscode/settings.json` in the repo with:

```json
{
	"python.defaultInterpreterPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe"
}
```

Note: VS Code will also detect virtual environments automatically if they are present in the workspace.

## Performance / Windows Defender note

If pip installs are very slow on Windows it's often because Windows Defender (MsMpEng) is real-time scanning many small files created during install. A common low-risk mitigation is to add a Defender exclusion for the project venv directory (requires admin):

```powershell
# Run as Administrator to add exclusion for the project venv and to the pip cache directory
Add-MpPreference -ExclusionPath "C:\\repositories\\adv-ai\\.venv"
Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\pip\Cache"
```

After finishing installs you can remove the exclusion with `Remove-MpPreference -ExclusionPath "C:\\repositories\\adv-ai\\.venv"`.

Keep security policies in mind (corporate machines may block changes).

## Working with Jupyter Notebooks (Git Version Control)

This repository uses `nbstripout` to keep notebooks clean in git. Cell outputs, metadata, and checkpoints are automatically excluded from version control. This prevents:

- Large diffs from regenerated plots and outputs
- Merge conflicts from auto-generated metadata
- Repository bloat from cell execution artifacts

### First-time setup (one-time per machine)

After installing dependencies:

```powershell
# Activate the venv
. .\.venv\Scripts\Activate.ps1

# Install/configure nbstripout globally
nbstripout --install --global
```

Or, to install for this repo only (no global config):

```powershell
nbstripout --install
```

This configures git to automatically strip cell outputs from `.ipynb` files before committing.

### Running notebooks

Open and edit notebooks in Jupyter or VS Code normally:

```powershell
# Activate the venv
. .\.venv\Scripts\Activate.ps1

jupyter notebook
# or
jupyter lab
```

Cell outputs are stored locally but won't be committed to git. To manually clear outputs:

```powershell
# Clear outputs from all notebooks
nbstripout notebooks/*.ipynb
```

### How it works

- `.gitattributes` configures git to use the `nbstripout` filter for `.ipynb` files.
- When you stage/commit a notebook, outputs and metadata are automatically removed.
- When you pull, the clean notebook is checked out.
- Notebooks regenerate outputs when you run them locally.

### Best practices

1. **Commit the clean `.py` scripts** (in `decision_trees/`, `knn/`, etc.) as the source of truth.
2. **Notebooks are for exploration/documentation** — regenerate their outputs by running cells.
3. **Keep meaningful notebook sections** with markdown explanations.
4. **Avoid large data in notebooks** — reference external data files or download via scripts.

## Data

Datasets are downloaded from Kaggle (or loaded via `sklearn.datasets`).
See `data/README.md` for dataset-specific instructions.

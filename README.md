# AI vs. Human Text Detector

I built this individual project as a Computer Science student at **Texas Tech University** to explore a practical question: can machine learning recognize differences between human-written and AI-generated text?

## Overview
I worked through the text classification process, from cleaning text and preparing features to comparing models and building a simple interface for trying predictions. My goal was to understand how different models approach the same problem and make the results easier to explore.

The training notebook compares six model families: SVM (LinearSVC), decision tree, AdaBoost, feedforward neural network, LSTM, and CNN. The Streamlit app serves the three available traditional models with a model selector.

**Tools:** Python, scikit-learn, TF-IDF, TensorFlow/Keras, pandas, Streamlit.

## My contribution
- Prepared text for modeling and used TF-IDF to turn writing into numerical features.
- Explored six model families in a training notebook and compared classification metrics.
- Saved the SVM, decision tree, and AdaBoost models for reuse.
- Built a Streamlit app that lets users enter text and select one of the three saved models.

## What I learned
This project strengthened my skills in Python, text preprocessing, model evaluation, and turning a notebook workflow into an application. It also taught me to question benchmark results and recognize the limits of automated authorship predictions.

## Run the app
Use Python 3.11 or 3.12. From this repository folder:
```bash
python -m venv .venv
```
Activate the environment on Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
Or on macOS/Linux:
```bash
source .venv/bin/activate
```
Install and launch:
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Included files
| File or folder | Purpose |
| --- | --- |
| `app.py` | Streamlit interface and model selector |
| `inference.py` | Shared preprocessing, artifact loading, and predictions |
| `models/` | Existing SVM, decision tree, AdaBoost, and TF-IDF artifacts |
| `notebooks/train_models.ipynb` | Six-model training and evaluation workflow |
| `data/README.md` | Dataset requirements |

## Training
Install `requirements-training.txt`, place the labeled Excel dataset in `data/`, and launch `python -m jupyterlab`. Open `notebooks/train_models.ipynb` with the kernel working directory set to `notebooks/`, then run the cells in order. Deep model training can take considerable time and memory.

The notebook reports accuracy, precision, recall, F1-score, and confusion matrices. Decision tree parameters are tuned with GridSearchCV. Dataset size and validated accuracy are not asserted here because a fresh training/evaluation run has not been completed for this package.

## Preparation notes
This upload version was prepared from the original project files. The app now uses matching text cleaning, paths relative to its source file, and avoids calling `predict_proba` on LinearSVC. The notebook splits text before fitting TF-IDF and the sequence tokenizer; its old outputs were cleared. Included artifacts are the original trained models, not models retrained using these notebook changes. The historical notebook fitted feature extraction before splitting, so reevaluation is needed before quoting benchmark results.

The original artifacts record scikit-learn 1.6.1, which is pinned in the requirements. Deep-learning weights and training data are not included. The app does not claim an external LLM API integration.

## Limitations
Classification depends on the training data and may fail on unfamiliar topics, short text, edited AI writing, or other languages. A predicted label is not proof of authorship. Tree/AdaBoost probabilities are model outputs, not verified reliability; LinearSVC has no probability output. Only load serialized model files from trusted sources.

## Author
Joshua-Bernard Takere · [GitHub](https://github.com/joshuabernard744-max)

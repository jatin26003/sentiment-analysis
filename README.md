# IMDB Sentiment Analysis using Deep Learning (RNN / LSTM)

Classifies IMDB movie reviews as **Positive** or **Negative** using NLP and a
Bidirectional LSTM (a type of RNN) built with TensorFlow/Keras.

## Project Files

| File | Description |
|---|---|
| `IMDB_RNN_LSTM.ipynb` | Main notebook (preprocessing, training, evaluation, prediction) |
| `IMDBDataset.csv` | Dataset (50,000 reviews, columns: `review`, `sentiment`). Put it in the same folder as the notebook |
| `sentiment_lstm_model.keras` | Saved model (created after running the notebook) |
| `tokenizer.pkl` | Saved tokenizer (created after running the notebook) |

## Requirements

- Python **3.9 – 3.12** (TensorFlow does not support very new Python versions; 3.11 is a safe choice)
- pip
- Jupyter Notebook or JupyterLab (or VS Code with the Jupyter extension)

## Libraries to Install

| Library | Purpose |
|---|---|
| `tensorflow` | Deep learning (Keras, Embedding, LSTM) |
| `pandas` | Loading and handling the dataset |
| `numpy` | Numerical operations |
| `nltk` | Stopword removal |
| `scikit-learn` | Train/test split, accuracy, confusion matrix |
| `matplotlib` | Plots |
| `seaborn` | Heatmap and histograms |
| `notebook` | To run the `.ipynb` file |

## Installation

### 1. (Recommended) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install everything with one command

```bash
pip install tensorflow pandas numpy nltk scikit-learn matplotlib seaborn notebook
```

Or save the list below as `requirements.txt` and run `pip install -r requirements.txt`:

```
tensorflow
pandas
numpy
nltk
scikit-learn
matplotlib
seaborn
notebook
```

### 3. Download the NLTK stopwords (one time)

The notebook does this automatically with `nltk.download('stopwords')`, but you
can also run it manually:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

## How to Run

1. Place `IMDBDataset.csv` in the same folder as the notebook.
2. Start Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open `IMDB_RNN_LSTM.ipynb`.
4. Run all cells from top to bottom (**Kernel → Restart & Run All**).

Training takes a few minutes on CPU and is faster with a GPU.

## Model Architecture

```
Embedding (20,000 words, 128 dims)
        ↓
Bidirectional LSTM (64 units)
        ↓
Dense (64, ReLU)
        ↓
Dropout (0.5)
        ↓
Dense (1, Sigmoid)  →  Positive / Negative
```

## Pipeline

1. Load the IMDB dataset
2. Clean text (lowercase, remove HTML and special characters, remove stopwords but keep negations like "not")
3. Split into train and test sets (80/20)
4. Tokenize and pad reviews to 200 words
5. Train the Bidirectional LSTM with early stopping
6. Evaluate (accuracy, classification report, confusion matrix, confidence scores)
7. Save the model and tokenizer
8. Predict sentiment on new reviews

## Troubleshooting

- **`No module named tensorflow`**: Your Python version may be too new. Use Python 3.11 and reinstall.
- **`FileNotFoundError: IMDBDataset.csv`**: Put the CSV in the same folder as the notebook.
- **Training is too slow**: Lower `MAX_LEN` to 150 or `batch_size` to 64 in the notebook.
- **NLTK `LookupError` for stopwords**: Run the download command from step 3.
- **Apple Silicon Mac**: Normal `pip install tensorflow` works on recent versions. For GPU, also install `tensorflow-metal`.

## Example Usage

```python
predict_sentiment("The movie was visually stunning and emotionally powerful")
# ('Positive', 97.12)
```

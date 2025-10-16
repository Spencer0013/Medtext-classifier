# Medtext Classifier 🧠📄

This project is a **sentence classification system for randomized controlled trial (RCT) abstracts**.
It classifies sentences into categories like **BACKGROUND**, **OBJECTIVE**, **METHODS**, **RESULTS**, and **CONCLUSIONS**.
The approach builds progressively from classical ML baselines to hybrid neural models combining token, character, and positional embeddings.

---

## 📑 Table of Contents

- [Demo](#-demo)
- [Project Overview](#-project-overview)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Results](#-model-results)
- [Future Work](#-future-work)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🚀 Demo

Below are some visual examples of preprocessing and model components used in the notebook.

image_group{"layout":"carousel","query":["PubMed RCT sentence classification diagram","text classification pipeline illustration","Conv1D text classification architecture diagram","token and character embedding fusion diagram"]}

---

## 📘 Project Overview

Medical research abstracts typically follow a structured format (e.g., background, objective, methods, results, conclusions).  
This project aims to automate **sentence-level classification** of these sections to make it easier to skim and structure literature.

**Key steps:**  
1. Load and preprocess PubMed-RCT dataset.  
2. Build multiple classification models: baseline to hybrid neural architectures.  
3. Evaluate and compare performance using standard metrics.  

**Dataset format:**  
```
### abstract_id
LABEL<TAB>Sentence text
...
```
where `LABEL` ∈ {BACKGROUND, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS}.

---

## 🛠 Tech Stack

- **Language:** Python 3.12  
- **Core Libraries:** TensorFlow (Keras), TensorFlow Hub, scikit-learn, pandas, numpy, matplotlib  
- **Optional:** Transformers (HuggingFace)

---

## 🧩 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/medtext-classifier.git
cd medtext-classifier

# (Optional) create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Or install manually:
```bash
pip install tensorflow tensorflow-hub tf-keras scikit-learn pandas numpy matplotlib transformers
```

---

## ▶️ Usage

1. Place your dataset files (`train.txt`, `dev.txt`, `test.txt`) in the project root.  
2. Open and run `Notebook.ipynb` in a GPU-backed environment.  
3. Models and results will be generated sequentially.

Typical file format:
```
### 12345678
BACKGROUND	Hypertension is prevalent...
OBJECTIVE	We evaluate...
METHODS	We conducted a randomized...
RESULTS	Treatment A reduced...
CONCLUSIONS	Treatment A may...
```

**Run order in notebook:**  
- Data loading and preprocessing  
- Tokenizer and vectorizer setup  
- Model definitions  
- Model training  
- Evaluation and result comparison

image_group{"layout":"carousel","query":["tensorflow model training notebook screenshot","keras model summary terminal output","matplotlib accuracy curve","confusion matrix heatmap classification"]}

---

## 🗂 Project Structure

```
.
├── Notebook.ipynb            # All model training, evaluation & preprocessing code
├── train.txt                 # Training set (not included in repo)
├── dev.txt                   # Validation set (not included)
├── test.txt                  # Test set (not included)
├── README.md
└── requirements.txt
```

**Key functions:**  
- `get_lines(path)` — Read raw dataset lines  
- `preprocess_text_with_line_numbers(path)` — Extract `text`, `target`, `line_number`, `total_lines`  
- `split_chars(text)` — Character-level tokenization  
- `calculate_results(y_true, y_pred)` — Evaluation metrics

---

## 📊 Model Results

The notebook trains and evaluates six models. Performance (on validation set):

| model                        | accuracy   | precision | recall  | f1       |
|------------------------------|------------|-----------|---------|----------|
| baseline                     | 72.18%     | 0.719     | 0.722   | 0.699    |
| custom_token_embed_conv1d    | 78.72%     | 0.784     | 0.787   | 0.785    |
| pretrained_token_embed       | 71.77%     | 0.718     | 0.718   | 0.715    |
| custom_char_embed_conv1d     | 66.45%     | 0.660     | 0.665   | 0.656    |
| hybrid_char_token_embed      | 73.62%     | 0.735     | 0.736   | 0.733    |
| tribrid_pos_char_token_embed | **83.27%** | 0.832     | 0.833   | 0.832    |

image_group{"layout":"carousel","query":["classification metrics bar chart","F1 score plot model comparison","neural network training history loss accuracy","tensorflow confusion matrix visualization"]}

---

## 🧭 Future Work

- Fine-tuning pretrained Transformer encoders (e.g., BERT variants).  
- Deploying as a lightweight API for real-time classification.  
- Improving robustness to noisy abstracts.  
- Extending to other domains (e.g., clinical trial reports).

---

## 🙏 Acknowledgments

- [PubMed 200k RCT Dataset](https://arxiv.org/abs/1710.06071)  
- TensorFlow, Keras, scikit-learn teams  
- HuggingFace Transformers community

---

## 📜 License

This project is open source and available under the MIT License.

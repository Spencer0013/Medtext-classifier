#  MedText Classifier

A state-of-the-art **medical abstract classification system** built on the [PubMed RCT dataset](https://github.com/Franck-Dernoncourt/pubmed-rct).  
This project demonstrates a **complete NLP pipeline** — from data ingestion and preprocessing to multiple deep learning models for classifying scientific sentences into structured labels such as `BACKGROUND`, `OBJECTIVE`, `METHODS`, `RESULTS`, and `CONCLUSIONS`.

---

##  Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Training](#-model-training)
- [Evaluation & Results](#-evaluation--results)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

##  Overview

Medical abstracts often follow a standardized structure. This project builds an NLP pipeline to automatically classify each sentence of an abstract into its correct **section category**, enabling downstream applications like literature summarization, evidence synthesis, and clinical decision support.

The project is implemented in Python and leverages deep learning architectures such as Conv1D and LSTMs to achieve strong classification performance.

---

## Key Features

-  **Text Preprocessing**: Tokenization, sentence splitting, label encoding, padding/truncation.  
-  **Multiple Models**:
  - Baseline model with TF-IDF + Mutltinomial
  - Conv1D with token Embedding layers
  - Feature Extraction with pretrained token embediings (Universal sentence Encoder)
  - Conv1D with character embeddings
  - Pretrained (USE) token embed + Char embed with Bi-LSTM
  - Pretrained (USE) token embed + Char embed with Bi-LSTM plus postional embeddings
  - Pretrained token BERT (SOTA)  + Char embed with Bi-LSTM plus postional embeddings
-  **Robust Evaluation**: Accuracy, Precision, Recall, F1-score.
-  **Data Visualization**: Sentence length distribution, label frequencies.
-  **Reproducible pipeline**: Clean notebook, modular code design.

---

## Dataset

The project uses the **PubMed RCT 20k dataset**, a collection of randomized controlled trial abstracts where each line is annotated with one of the following labels:

- `BACKGROUND`  
- `OBJECTIVE`  
- `METHODS`  
- `RESULTS`  
- `CONCLUSIONS`

Dataset files:
```
train.txt
dev.txt
test.txt
```

> **Source**: [Franck-Dernoncourt/pubmed-rct](https://github.com/Franck-Dernoncourt/pubmed-rct)

---

##  Architecture

1. **Data Layer**: Load and parse text data from structured files.  
2. **Preprocessing Layer**:
   - Clean sentences
   - Encode labels (label + one-hot)
   - Tokenize and vectorize
3. **Modeling Layer**:
   - TF-IDF + Logistic Regression (baseline)
   - Conv1D + Embedding
   - Bi-LSTM architectures
4. **Evaluation Layer**:
   - Accuracy, Precision, Recall, F1-score
   - Visualization of results

---

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/medtext-classifier.git
cd medtext-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Example `requirements.txt`:
```
numpy
pandas
tensorflow
tf_keras
scikit-learn
matplotlib
seaborn
jupyter
```

---

##  Usage

Launch Jupyter Notebook to explore the pipeline:

```bash
jupyter notebook Notebook.ipynb
```


---

## Model Training

1. **Load Dataset**  
2. **Preprocess Sentences and Labels**  
3. **Vectorize Text (TextVectorization or TF-IDF)**  
4. **Train Model** 
5. **Evaluate on Dev/Test**

---

## Evaluation & Results

Metrics computed:
- Accuracy  
- Precision  
- Recall  
- F1-score  

![alt text](image.png)


---

## Tech Stack

- **Language**: Python 3.x  
- **Libraries**: TensorFlow, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn , Transformers
- **Notebook**: Jupyter  
- **Dataset**: PubMed RCT 20k

---


## Contributing

Contributions are welcome!

1. Fork the project  
2. Create a feature branch: `git checkout -b feature/your-feature`  
3. Commit changes: `git commit -m "Add feature"`  
4. Push to branch: `git push origin feature/your-feature`  
5. Open a Pull Request

---

## Author

This project was developed by Opeyemi Aina

## License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute with attribution.

---


## References

- [Franck-Dernoncourt/pubmed-rct](https://github.com/Franck-Dernoncourt/pubmed-rct)
- TensorFlow Documentation
- scikit-learn Documentation
- Academic Paper: [PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts](https://arxiv.org/abs/1710.06071)

---

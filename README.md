---
title: SkimLit-Style Abstract Classifier  
emoji: 🧪  
colorFrom: blue  
colorTo: green  
sdk: streamlit  
sdk_version: 1.39.0  
app_file: app.py  
pinned: false  
short_description: Predict roles/classes in medical abstracts with classic baselines and a BERT (model_6) system, plus a simple Streamlit predictor.
---

# SkimLit-Style Abstract Classifier

## Overview

**Problem in a sentence**  
The number of RCT papers is exploding, and abstracts that aren’t clearly structured slow researchers down. Sifting through “Background, Methods, Results, Conclusions” by hand is tedious and error-prone (see screenshot).  

**Solution in a sentence**  
Build an NLP pipeline that learns to classify text from medical abstracts—first at the sentence level (roles like *BACKGROUND*, *METHODS*, *RESULTS*, *CONCLUSIONS*), and then provide a lightweight app that predicts the overall abstract class using the final **`model_6`** BERT system.

### What this repo covers
- A complete workflow inspired by the SkimLit approach:
  - Data loading and preprocessing for **PubMed 20k RCT**.
  - Multiple modeling experiments (TF-IDF baselines → token/char CNNs → hybrids → **BERT**).
  - Saved final model (**`model_6`**) and a **Streamlit** app that predicts abstract class (prediction-only; no evaluation in the UI).
- Clear pointers to how to reproduce the experiments in the notebook and use the app for inference.

The screenshot (`/mnt/data/bc37c9a2-3a68-42c9-b21a-2ff23531cd44.png`) summarizes the problem, solution, and learning plan we followed.

---

## Dataset

**Source:** PubMed 20k RCT (numbers replaced with `@`) — a widely used dataset for sequential sentence classification in medical abstracts.

- **Total abstracts:** **20,000**
- **Split used:** **18,000 train / 2,000 dev / 2,000 test** (abstract-level; each abstract contains labeled sentences)
- **Sentence labels:** `BACKGROUND`, `OBJECTIVE`, `METHODS`, `RESULTS`, `CONCLUSIONS`
- **Preprocessing (notebook):**
  - Parse `.txt` files into `(abstract_id, label, sentence)` rows
  - Clean text (minimal; preserve biomedical tokens)
  - Encode labels with `LabelEncoder`
  - For transformer models: tokenize with **Hugging Face `bert-base-uncased`** to fixed length (`SEQ_LEN=128`)

> The notebook handles the splits and label encoding; the app consumes the already-saved model for prediction.

---

## Experiments

The notebook steps through a series of models, increasing in capacity and prior knowledge. The variable names below mirror the code:

1. **`baseline` — TF-IDF → Linear Classifier (e.g., MultinomialNB / Logistic)**
   - **What:** Classic bag-of-words with n-grams.
   - **Why:** Establish a quick, interpretable baseline.
   - **Notes:** No deep features; fast to train and a strong sanity check.

2. **`model_1` — `custom_token_embed_conv1d`**
   - **What:** Learnable token embeddings + 1D CNN stack.
   - **Why:** Capture local n-gram patterns with convolutional filters.
   - **Notes:** Pure token path; regularization via dropout.

3. **`model_2` — `pretrained_token_embed`**
   - **What:** Swap in pretrained token embeddings (vs. random init).
   - **Why:** Inject distributional prior to improve generalization, especially for rarer medical terms.

4. **`model_3` — `custom_char_embed_conv1d`**
   - **What:** Character-level embeddings + CNNs.
   - **Why:** Robust to misspellings, punctuation quirks, and OOV tokens common in clinical text.

5. **`model_4` — `hybrid_char_token_embed`**
   - **What:** Concatenate **token** and **character** paths.
   - **Why:** Blend subword robustness with token semantics.

6. **`model_5` — `tribrid_pos_char_token_embed`**
   - **What:** Add a simple **positional**/sequence-aware signal to the hybrid.
   - **Why:** Sentence roles in abstracts strongly follow order (e.g., *BACKGROUND* → *METHODS* → *RESULTS* → *CONCLUSIONS*).

7. **`model_6` — `tribrid_bert_classifier` (final)**
   - **What:** A **BERT** path built with Hugging Face `bert-base-uncased` (TensorFlow) producing a pooled representation, projected to a 128-dim token embedding and integrated with the existing architecture. Implemented with custom Lambda layers:
     - `_hf_tokenize_layer` (wraps HF tokenization under `tf.numpy_function`)
     - `_squeeze_if_needed` (utility for shape handling)
   - **Training details (from notebook code):**
     - `CategoricalCrossentropy(label_smoothing=0.2)`
     - `keras.optimizers.Adam()`
     - `SEQ_LEN = 128`
     - Backbones instantiated via `TFBertModel.from_pretrained(...)`
   - **Why:** Modern transformer representations capture long-range context and biomedical semantics better than shallow features.

### Evaluation results

The notebook computes evaluation tables (accuracy/precision/recall/F1) for each model and aggregates them into `all_model_results`, which includes:

```
baseline
custom_token_embed_conv1d
pretrained_token_embed
custom_char_embed_conv1d
hybrid_char_token_embed
tribrid_pos_char_token_embed
tribrid_bert_classifier
```

> Exact numeric scores are presented in the notebook’s outputs (the app intentionally **does not** show metrics; it’s prediction-only). The observed trend is the expected one: classical baselines → token/char CNNs → hybrids → **BERT (`model_6`)** improving the overall classification quality.

---

## Application (prediction-only)

We ship a small **Streamlit** app to make predictions with the **final BERT system (`model_6`)**:

- Loads a **SavedModel** directory (default: `./save_model`) created in the notebook:  
  `model_6.save("save_model")`
- Reconstructs the tokenizer and custom functions so the graph loads cleanly.
- **Two modes:**
  - **Single abstract:** paste text, get predicted class + confidence.
  - **Batch:** upload a CSV with one `abstract` column; download predictions.

> The app does not evaluate or fine-tune; it reuses the trained weights for fast inference.

### Run locally

```bash
# minimal runtime stack
pip install -U streamlit tensorflow tf-keras transformers pandas numpy

# start the app
streamlit run app.py
```

If your model folder is zipped, upload it in the sidebar; otherwise point the path to `save_model`.

---

## Project Structure

```
.
├── app.py                # Streamlit prediction app (uses ONLY model_6)
├── Notebook.ipynb        # Full experiment pipeline and training code
├── save_model/           # (not tracked) Keras SavedModel produced by model_6
├── requirements.txt      # (optional) consolidated dependencies
└── README.md             # this document
```

---

## Reproducibility notes

- **Randomness:** Set seeds where practical; deep models may still vary slightly.
- **Hardware:** Transformer experiments train faster on GPU.
- **Data:** Ensure you’re using the **20k** split (`train/dev/test`), not the 200k variant.
- **Sequence length:** Notebook uses `SEQ_LEN=128`; keep this consistent when exporting and serving.

---

## Limitations & future work

- **Sentence vs. abstract scope:** The experimental focus is sentence-role modeling; the shipped app performs **abstract-level** prediction with the final system for simplicity. Extending the UI to show **per-sentence** labels (and highlight spans) would better mirror the research goal.
- **Domain shift:** Biomedical terminology is nuanced; consider domain-specific BERT variants (e.g., BioBERT, PubMedBERT) for further gains.
- **Structure awareness:** Adding explicit sequential modeling (e.g., CRF over sentence roles) could improve consistency across an abstract.

---

## Acknowledgments

- SkimLit concept and dataset preparation pipelines from prior literature on **PubMed 20k RCT**.
- Hugging Face Transformers for the BERT backbone and tokenizer.
- TensorFlow / Keras for modeling; Streamlit for app delivery.

---

## Author

Built by **Dagogo Orifama**.  
For questions or collaboration:

- GitHub: <https://github.com/DagogoOrifama>  
- LinkedIn: <https://www.linkedin.com/in/dagogoorifama/>  

> If you want this README mirrored on a Space, keep the YAML block at the top (`app_file: app.py`) so it deploys smoothly.

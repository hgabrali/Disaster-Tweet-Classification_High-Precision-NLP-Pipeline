# Technical Project Report: Disaster Tweet Classification

## 📋 Project Overview
**Project Title:** High-Precision NLP Pipeline for Real-Time Crisis Identification  
**Frameworks:** `PyTorch`, `HuggingFace Transformers`, `Scikit-Learn`

---

## 1. 🎯 Business Objective & Problem Framing
The primary goal of this initiative is to distinguish between tweets describing actual disaster events (e.g., "Earthquake strikes the city") and metaphorical or non-critical mentions (e.g., "This pizza is the bomb").

* **Metric of Success:** The **F1-Score** is prioritized over Accuracy due to the critical nature of false negatives in disaster scenarios.
* **Engineering Challenge:** Twitter data is notoriously noisy, containing URLs, HTML artifacts, and colloquialisms that require specialized preprocessing.

---

## 2. 🏗️ Data Architecture & Preprocessing (Scrubbing)
The pipeline implements a **"Clean-for-BERT"** strategy, which is more surgical than traditional stop-word removal.

<img width="292" height="173" alt="image" src="https://github.com/user-attachments/assets/43e7875d-47ef-45ee-9b67-5e698e7a4a72" />

<img width="1320" height="501" alt="image" src="https://github.com/user-attachments/assets/4ab84578-c00b-4d90-9757-71c9893768a9" />


### 🛡️ Leakage Prevention
A **stratified split** (10% Hold-out, 90% Development) was implemented before applying transformations. This ensures the TF-IDF vectorizer and Transformer tokenizer do not see the test distribution during training, maintaining the integrity of the evaluation.

### 🧹 Cleaning Protocol
* **URL and HTML tag removal:** Executed via Regex to eliminate non-textual noise.
* **Whitespace normalization:** Ensuring consistent spacing for tokenizer efficiency.
* **Case folding (Lowercase):** Reduces vocabulary sparsity and ensures uniform token mapping.

---

## 3. 🧠 Modeling Strategy: Comparison & Benchmarking

### Model A: TF-IDF + Logistic Regression (The Baseline)
A classical statistical approach using term frequency-inverse document frequency.

<img width="455" height="255" alt="image" src="https://github.com/user-attachments/assets/43f6bb9f-30ee-4e3e-a2ae-1f3423250b40" />


| Pros | Cons |
| :--- | :--- |
| **High Interpretability:** Easy to extract feature importance (which words signify a disaster). | **No Contextual Awareness:** Treats "Fire" in "Fire in the building" the same as "Fire track on Spotify." |
| **Low Latency:** Inference is nearly instantaneous; minimal CPU requirements. | **Sparse Representation:** Struggles with synonyms or misspelled words. |

### Model B: DeBERTa-v3-Small (Advanced Transformer)
A modern Transformer architecture using "Decoding-enhanced BERT with disentangled attention."

<img width="428" height="200" alt="image" src="https://github.com/user-attachments/assets/e939197b-e1b3-4260-9fd7-4d6ab0953fb5" />


| Pros | Cons |
| :--- | :--- |
| **Deep Semantic Understanding:** Captures word relationships and context via attention mechanisms. | **Resource Intensive:** Requires GPU acceleration (T4 used) and significantly higher training time. |
| **Superior Generalization:** Better at handling out-of-vocabulary (OOV) slang common on Twitter. | **Black-box Nature:** Harder to explain specific classification decisions compared to Logistic Regression. |

---

## 4. 📊 Evaluation & Performance Summary
The transition from Baseline to DeBERTa represents a significant jump in the ROC-AUC and F1-Score.

<img width="1725" height="648" alt="image" src="https://github.com/user-attachments/assets/81dd6b3c-24cc-4f85-b4c7-7156b830dcd5" />


### Comparative Performance Matrix
| Metric | Baseline (LogReg) | Advanced (DeBERTa-v3) | Delta (Δ) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | Baseline Value | Model Value | +Improvement |
| **F1-Score** | Baseline Value | Model Value | +Improvement |
| **AUC** | Baseline Value | Model Value | +Improvement |





### 🔍 Error Analysis (Interpretation)
The model's failure points (Misclassified Examples) often stem from:

<img width="651" height="480" alt="image" src="https://github.com/user-attachments/assets/9e1802c9-0463-4dbe-b9c5-f3202342a574" />


<img width="832" height="471" alt="image" src="https://github.com/user-attachments/assets/86460053-6d16-4674-b12c-a55bbd160502" />

1.  **Sarcasm/Irony:** Tweets that use disaster terminology in a joking context.
2.  **Ambiguous Keywords:** Words like "Ablaze" or "Siren" used in song lyrics vs. actual emergencies.

---

## 5. 🛠️ Engineering Key Takeaways
* **Transformer Dominance:** In short-form text (Tweets), the ability of DeBERTa to understand word positions via **Disentangled Attention** provides a definitive edge over the Bag-of-Words approach.
* **Pipeline Robustness:** The use of `DataCollatorWithPadding` ensures that dynamic batching is efficient, reducing memory overhead during training.
* **Deployment Readiness:** Saving the model in the `PreTrained` format allows for seamless integration with API frameworks like **FastAPI** or **Flask**.

---

## 6. 🚀 Recommended Next Steps

1.  **Hyperparameter Optimization:** Implement `Optuna` to fine-tune the learning rate ($2e-5$) and weight decay ($0.01$) more precisely.
2.  **Ensemble Learning:** Create a weighted ensemble between the Baseline and DeBERTa. Sometimes the simplicity of TF-IDF can act as a "sanity check" for the complex Transformer.
3.  **Data Augmentation:** Use **Back-Translation** (translating to French and back to English) to increase the size of the 'Disaster' class.

---

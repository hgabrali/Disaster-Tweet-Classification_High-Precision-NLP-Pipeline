# 🚨 Disaster Tweet Classification: High-Precision NLP Pipeline

This repository implements an end-to-end **State-of-the-Art (SOTA)** NLP pipeline designed to classify whether a tweet describes a real-world disaster or utilizes hyperbolic language metaphorically. This project specifically addresses the challenge of **semantic ambiguity** (e.g., *"This song is fire"* vs. *"The forest is on fire"*) through a robust six-phase engineering roadmap.

---

## 🏗 Project Roadmap & Architecture

### Phase 1: Data Acquisition & Discovery
* **Ingestion**: The core dataset comprises approximately 7,600 labeled tweets. In a production environment, the architecture is designed to ingest data via the **Twitter/X API v2** into a **BigQuery** or **AWS S3** data lake.
* **Exploratory Data Analysis (EDA)**: Initial analysis reveals a class distribution of **57% (Non-Disaster)** versus **43% (Disaster)**. Text lengths peak at 130–140 characters, consistent with microblogging constraints.
* **Ground Truth**: To mitigate label noise, we implement **Multi-Annotator Agreement (Cohen’s Kappa)** to ensure that "Gold Standard" labels remain consistent even across highly ambiguous samples.

### Phase 2: Text Preprocessing & Normalization
To maximize model stability, we minimize linguistic noise through a custom normalization pipeline:
* **Regex Cleaning**: Systematic stripping of URLs, HTML entities, and user handles.
* **Tokenization & Lemmatization**: Utilizing **spaCy** for context-aware lemmatization (e.g., reducing "burning" and "burnt" to the root "burn") to preserve semantic integrity better than crude stemming.



### Phase 3: Text Representation (Vectorization)
We transition from statistical baselines to dense contextual embeddings:
* **Baseline**: **TF-IDF with Bi-grams** to capture localized phrase-level statistics.
* **SOTA**: **Transformer Encoders (RoBERTa-base / BERT-tweet)**. Unlike static embeddings (Word2Vec), these generate **Contextualized Embeddings**, allowing the model to distinguish word meanings based on surrounding tokens.

### Phase 4: Model Development & Fine-Tuning
* **Architecture Selection**: A fine-tuned **RoBERTa** model. We employ **Transfer Learning** by utilizing a pre-trained encoder and appending a specialized classification head (Linear Layer + Dropout).
* **Hyperparameter Optimization**: Utilizing **Optuna** for Bayesian search over learning rates ($2e-5$ to $5e-5$) and weight decay.
* **Training Strategy**: **Cross-Entropy Loss** with **Early Stopping** to prevent overfitting on the relatively small corpus.



### Phase 5: Evaluation & Error Analysis
* **Primary Metric**: **F1-Score**. This is prioritized over accuracy to balance the critical trade-off between **False Positives** ("crying wolf") and **False Negatives** (missing a real-time disaster).
* **Explainability**: Implementation of **SHAP (SHapley Additive exPlanations)** or **LIME** to visualize feature importance. This ensures the model is not over-relying on isolated keywords regardless of context.
* **Confusion Matrix**: Systematic analysis of **"Metaphorical Failures"** to guide targeted data augmentation.

### Phase 6: Deployment & Monitoring
* **Inference Optimization**: To minimize latency, we utilize **Dynamic Quantization (FP32 → INT8)** and export the final model to the **ONNX** format.
* **Containerization**: The service is wrapped in **FastAPI**, containerized via **Docker**, and orchestrated using **Kubernetes (K8s)**.
* **Drift Monitoring**: Automated tracking for **Data Drift** (evolution in Twitter slang) and **Concept Drift** (emergence of new disaster types, such as pandemic-specific terminology).

---

## 🚀 Pro-Tip: Multi-Modal Integration
While text is the primary feature, a production-grade iteration of this project leverages keyword and location metadata. By utilizing a **Multi-Modal Architecture**, we concatenate **RoBERTa text embeddings** with **Categorical Embeddings** for keywords, significantly boosting the F1-Score in localized disaster scenarios.

---

## 🛠 Tech Stack
* **Language**: Python 3.9+
* **Modeling**: HuggingFace Transformers, PyTorch, Scikit-learn
* **Tuning**: Optuna
* **Deployment**: FastAPI, Docker, ONNX
* **Explainability**: SHAP / LIME

---

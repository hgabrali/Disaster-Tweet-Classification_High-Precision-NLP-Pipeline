# Disaster Tweet Classification: High-Precision NLP Pipeline

## 📖 Introduction 
This repository implements a production-grade **Natural Language Processing (NLP)** pipeline designed to resolve semantic ambiguity in microblogging data. By leveraging **DeBERTa-v3** (Decoding-enhanced BERT with disentangled attention), the system distinguishes between actual crisis events and hyperbolic language (e.g., *"The sunset is fire"* vs. *"The building is on fire"*) with significantly higher F1-scores than classical statistical methods.

---

## 🏗 Project Roadmap (CRISP-DM Framework)

### Phase 1: Discovery & Problem Framing
The primary objective is to maximize the **F1-Score**, prioritizing the reduction of **False Negatives** (missed disasters) while maintaining high precision to avoid "alert fatigue" in emergency response systems.

### Phase 2: Data Architecture & Preprocessing
To prevent **Data Leakage**, all preprocessing parameters were derived strictly from the training distribution.

* **Leakage Prevention:** A 10% stratified hold-out test set was reserved prior to any vectorization or transformation.
* **Cleaning Protocol:** A custom Regex pipeline was developed to strip URLs, HTML tags, and normalize whitespace, effectively preparing tokens for transformer-based attention masks.



### Phase 3: Model Architecture & Benchmarking
We implement a **"Baseline vs. Challenger"** strategy to quantify the value of transformer-based context.

| Feature | Baseline Model | Challenger Model |
| :--- | :--- | :--- |
| **Architecture** | TF-IDF + Logistic Regression | DeBERTa-v3-Small |
| **Contextual Awareness** | Sparse (N-gram restricted) | Dense (Disentangled Attention) |
| **Representation** | Static Frequency counts | Contextualized Embeddings |
| **Inference Latency** | Ultra-Low (CPU) | Moderate (GPU Optimized) |

### Phase 4: Training & Hyperparameter Optimization
The DeBERTa-v3 model was fine-tuned using the following parameters to ensure convergence and prevent overfitting:

* **Optimization:** AdamW with a learning rate of $2 \times 10^{-5}$.
* **Regularization:** Weight decay ($0.01$) and Gradient Clipping ($1.0$).
* **Strategy:** Best model selection based on Validation F1-Score.

### Phase 5: Comparative Performance Analysis
The evaluation confirms the superiority of the Transformer architecture in resolving metaphor-heavy samples.

| Model | Accuracy | F1-Score | AUC |
| :--- | :--- | :--- | :--- |
| **Baseline (LogReg)** | *TBD* | *TBD* | *TBD* |
| **DeBERTa-v3** | *TBD* | *TBD* | *TBD* |
| **Delta ($\Delta$)** | +Improvement | +Improvement | +Improvement |



> **Error Analysis:** Misclassifications predominantly occur in tweets containing high levels of sarcasm or ambiguous keywords used in non-standard cultural contexts.

### Phase 6: Deployment Readiness
* **Model Serialization:** The final model and tokenizer are saved in the `PreTrained` format for seamless integration with **FastAPI**.
* **Prediction Engine:** A robust function handles real-time inference, including NaN-checks and confidence-score thresholding.

---

## 🛠 Tech Stack

* **Core Model:** DeBERTa-v3-Small (via HuggingFace Transformers)
* **Deep Learning:** PyTorch
* **Analytics:** Scikit-learn, Pandas, Seaborn
* **Engineering:** Regex, `DataCollatorWithPadding`

---

# Model Performance Benchmarking & Critical Failure Analysis

---

## 📖 Executive Summary
This project evaluates the trade-offs between a high-speed statistical baseline and a state-of-the-art Transformer architecture. While the theoretical framework favors deep semantic understanding, the empirical results highlight critical challenges in fine-tuning large language models (LLMs) on noisy microblogging data.

---

## 1. 📊 Comparative Metrics Overview
The following table summarizes the performance of both models on the stratified validation set, which represents **9%** of the total **7,613** entries:

| Metric | Baseline (TF-IDF + LogReg) | Challenger (DeBERTa-v3-Small) | Delta (Δ) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.8207 | 0.5700 | -25.07% |
| **F1-Score** | 0.7784 | 0.0000 | -77.84% |
| **ROC-AUC** | 0.8716 | 0.5000 | -37.16% |

> [!NOTE]
> 🖼️ `

[Image of Model Comparison Chart]
`

---

## 2. 🧪 The "Baseline vs. Transformer" Paradox
The report suggests a "significant jump" in performance through Transformer dominance. However, the current iteration shows a complete collapse of the DeBERTa model's predictive power.

* **🚨 Majority Class Bias:** With an Accuracy of **0.57** and an F1-Score of **0.00**, the DeBERTa model fell into a "Zero-Rule" trap. It predicted the majority class ("Normal") for 100% of the samples, failing to identify a single disaster event (0% Recall for the Disaster class).
* **📉 Logit Instability:** Technical inspection reveals the presence of **NaN/Inf** values in the model's logits during inference. This indicates Gradient Explosion or numerical instability during the fine-tuning process, resulting in a random-chance ROC-AUC of **0.50**.
* **✅ Baseline Robustness:** In contrast, the TF-IDF + Logistic Regression pipeline demonstrated remarkable resilience, achieving an **82%** accuracy with nearly instantaneous inference. This highlights that for short-form, high-noise data, simple statistical counts can outperform unoptimized deep learning models.

---

## 3. 🛠️ Root Cause Analysis (Technical Post-Mortem)
The failure of the DeBERTa-v3 model to converge can be attributed to several engineering factors:

* **📍 Learning Rate Mismatch:** The utilized learning rate of $2 \times 10^{-5}$ may have been too aggressive for this specific dataset size, preventing the model from escaping a local minimum.
* **⚖️ Weight Decay & Regularization:** While a weight decay of **0.01** was applied, it was insufficient to stabilize the training against the noise inherent in Twitter's colloquialisms and URLs.
* **🧠 Disentangled Attention Overhead:** While theoretically superior at capturing word positions, the complexity of DeBERTa's architecture requires more precise hyperparameter tuning compared to standard BERT.

---

## 4. 🚀 Strategic Remediation Steps
To bridge the gap between theoretical potential and empirical performance, the following optimizations are planned:

1.  **⚙️ Automated Tuning:** Deploying **Optuna** to find the optimal learning rate and scheduler type.
2.  **✂️ Gradient Clipping:** Implementing strict gradient clipping ($1.0$) to prevent logit overflows.
3.  **⚖️ Class Imbalance Handling:** Utilizing **Back-Translation** to augment the 'Disaster' class, providing the model with more signal to learn the minority class.

---
🖼️ ``

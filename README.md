# ⚖️ BiasGuard: AI Fairness Auditor

**BiasGuard** is an open-source auditing tool built with Streamlit, Pandas, and the Google GenAI SDK. It enables machine learning engineers, data scientists, and compliance teams to evaluate tabular datasets for algorithmic disparate impact and generate LLM-powered mitigation reports.

---

## 📌 Overview

Algorithmic bias occurs when machine learning models or underlying training datasets systematically disadvantage specific demographic groups. 

BiasGuard automates the detection of disparate impact based on the **EEOC Four-Fifths (80%) Rule**:

$$\text{Disparate Impact Ratio (DIR)} = \frac{P(\hat{Y} = 1 \mid D = \text{Unprivileged})}{P(\hat{Y} = 1 \mid D = \text{Privileged})}$$

* **$\text{DIR} \ge 0.80$**: Passes the legal standard for non-discriminatory outcome distribution.
* **$\text{DIR} < 0.80$**: Flags adverse/disparate impact against the unprivileged demographic group.

---

## 🚀 Key Features

* **Dynamic CSV Ingestion & Preprocessing:** Upload any CSV file with automatic string whitespace sanitization and tabular previews.
* **Custom Metric Configuration:** Dynamically map target outcomes, protected attributes, and demographic classes directly from dataset schema.
* **Disparate Impact Metric Engine:** Real-time calculation of group-level favorable rates and overall Disparate Impact Ratio (DIR).
* **Automated AI Executive Summaries:** Integrates `gemini-2.5-flash` via the modern `google-genai` SDK to convert statistical audit results into technical debiasing roadmaps (re-weighting, adversarial debiasing, synthetic sampling).

---

## 🛠️ Tech Stack

* **Frontend/Framework:** [Streamlit](https://streamlit.io/)
* **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
* **AI Analysis:** [Google GenAI SDK (`google-genai`)](https://github.com/googleapis/python-genai) — Model: `gemini-2.5-flash`

---

## 📂 Project Structure

```text
biasguard/
├── .streamlit/
│   └── secrets.toml          # Local API credentials (ignored by git)
├── app.py                    # Core Streamlit application
├── requirements.txt          # Python dependencies
├── LICENSE
└── README.md

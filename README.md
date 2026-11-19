# Olist E-Commerce Analytics & Predictive Modeling

An end-to-end Data Science project exploring the Brazilian E-Commerce Public Dataset by Olist. This project combines data analysis, SQL, statistical testing, machine learning, explainable AI, customer segmentation, and NLP to uncover business insights and predict late deliveries.

**Repository:** [Olist-E-commerce-Analysis](https://github.com/Hulash-shah/Olist-E-commerce-Analysis)

---

## Project Overview

The Olist dataset contains approximately 100,000 orders made between 2016 and 2018 through a Brazilian e-commerce marketplace.

The goal of this project is to transform raw e-commerce data into actionable business insights by investigating customer behavior, sales performance, delivery operations, and customer satisfaction.

The project follows an end-to-end workflow, from data preparation and exploratory analysis to predictive modeling and interactive visualization.

## Key Objectives

- Integrate multiple relational datasets into a unified order-level dataset.
- Explore sales trends, customer behavior, product categories, and delivery performance.
- Perform SQL-based business analysis and statistical hypothesis testing.
- Develop machine learning models to predict late deliveries.
- Interpret model predictions using SHAP explainability.
- Segment customers using K-Means clustering.
- Analyze customer review sentiment using Natural Language Processing.
- Build an interactive dashboard using Streamlit.

## Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Statistics | SciPy |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Database | SQLite, SQL |
| Machine Learning | Scikit-learn, XGBoost |
| Explainable AI | SHAP |
| NLP | TF-IDF, Logistic Regression |
| Dashboard | Streamlit |
| Development | Jupyter Notebook, Git, GitHub |

## Project Workflow

### 1. Data Preparation & Integration

- Worked with nine related Olist datasets.
- Handled missing values and inconsistent records.
- Converted timestamps and engineered relevant features.
- Aggregated order items, payments, and reviews before joining datasets to prevent duplicate orders.
- Calculated delivery duration, delivery delay, shipping distance, freight ratio, and other features.

### 2. Exploratory Data Analysis (EDA)

Investigated:

- Monthly revenue and order trends.
- Top-performing product categories.
- Customer and seller locations.
- Delivery duration and late delivery patterns.
- Relationship between delivery performance and customer reviews.

### 3. SQL Business Analysis

Used SQLite to perform business-oriented queries, including:

- Revenue by product category.
- Late delivery rates by customer state.
- Monthly revenue trends.
- Customer review comparisons between late and on-time orders.

### 4. Statistical Analysis

Applied statistical techniques to investigate relationships within the data:

- Mann–Whitney U test: Customer review scores versus delivery performance.
- Chi-square test: Relationship between same-state shipping and late deliveries.
- Spearman correlation: Relationship between shipping distance and delivery duration.

### 5. Late Delivery Prediction

Developed and compared three classification algorithms:

- Logistic Regression
- Random Forest
- XGBoost

The models were evaluated using ROC-AUC, PR-AUC, Precision, Recall, and F1-score.

**XGBoost achieved the best overall PR-AUC and ROC-AUC among the evaluated models.**

### 6. Explainable AI with SHAP

Used SHAP (SHapley Additive exPlanations) to interpret the XGBoost model and understand which features contributed most to late-delivery predictions.

Important features included purchase month, estimated delivery time, shipping distance, customer location, and approval time.

### 7. Customer Segmentation

Applied K-Means clustering to group customers based on purchasing behavior and related characteristics.

Identified four customer segments:

- High-value one-time customers
- Dissatisfied customers
- Low-value one-time customers
- Repeat buyers

### 8. NLP – Review Sentiment Analysis

Developed a sentiment classification model using:

- TF-IDF vectorization
- Unigrams and bigrams
- Logistic Regression

Customer reviews were classified using their associated review scores, with scores of 4–5 treated as positive and 1–2 as negative. Neutral reviews were excluded.

The analysis also explored the relationship between delivery delays and negative customer feedback.

### 9. Interactive Streamlit Dashboard

Built an interactive dashboard with four sections:

1. **Overview:** Business KPIs, revenue trends, product categories, and delivery performance.
2. **Late-Delivery Model:** Model comparison, ROC/PR curves, SHAP explanations, and an interactive order prediction interface.
3. **Customer Segments:** Customer segment profiles and visualizations.
4. **Review Sentiment:** NLP performance metrics, sentiment insights, and a review sentiment tester.

---

## Key Results & Findings

### Dataset Summary

| Metric | Result |
|---|---:|
| Original orders | 99,441 |
| Integrated master dataset | 98,666 |
| Orders used for delivery modeling | 96,427 |
| Late delivery rate | 8.07% |

### Machine Learning Model Performance

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7140 | 0.1849 | 0.1869 | 0.4374 | 0.2619 |
| Random Forest | 0.7944 | 0.2954 | 0.2996 | 0.4682 | 0.3654 |
| **XGBoost** | **0.7990** | **0.3062** | **0.3108** | **0.4701** | **0.3742** |

### Customer Satisfaction Insights

- Late-delivered orders had an average review score of **2.56**, compared with **4.29** for on-time orders.
- Negative reviews accounted for **54.1%** of late orders with review scores, compared with **9.2%** of on-time orders with review scores.
- The analysis found a statistically significant difference in review scores between late and on-time deliveries.

### Customer Segmentation

- Identified four distinct customer segments.
- Repeat buyers represented approximately **3.0%** of the segmented customer base.
- Customer groups showed differences in purchasing value, frequency, review scores, and freight burden.

### NLP Performance

| Metric | Result |
|---|---:|
| Reviews with text | 35,491 |
| ROC-AUC | 0.9753 |
| Macro F1-Score | 0.9164 |
| Positive review share | 74.0% |

**Important finding:** Negative sentiment appeared in 74.7% of late-order review comments, compared with 20.1% of on-time-order review comments.

---

## Repository Structure

```text
Olist-E-commerce-Analysis/
│
├── data/
│   ├── olist_master.csv.gz
│   ├── olist_delivered.csv.gz
│   ├── reviews_sentiment.csv.gz
│   └── customer_segments.csv
│
├── models/
│   ├── xgb_late.json
│   └── xgb_meta.json
│
├── notebooks/
│   ├── brazilian-e-commerce.ipynb
│   └── brazilian-e-commerce_EDA.ipynb
│
├── outputs/
│   ├── figures/
│   ├── metrics.json
│   ├── model_comparison.csv
│   └── SQL analysis outputs
│
├── sql/
│   └── queries.sql
│
├── app.py
├── pipeline.py
├── requirements.txt
└── README.md
```

*Some generated files inside `data/` and `outputs/` are created when the pipeline runs.*

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Hulash-shah/Olist-E-commerce-Analysis.git
cd Olist-E-commerce-Analysis
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the Raw Dataset

Download the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

Place the nine original CSV files inside a folder named `raw/` in the project root.

The expected filenames are:

```text
raw/
├── olist_customers_dataset.csv
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_products_dataset.csv
├── olist_sellers_dataset.csv
├── olist_geolocation_dataset.csv
└── product_category_name_translation.csv
```

### 5. Run the Data Science Pipeline

```bash
python pipeline.py --raw raw
```

The pipeline performs data integration, SQL analysis, EDA, statistical testing, model training, SHAP analysis, customer segmentation, and sentiment analysis.

Generated datasets, model artifacts, and analysis outputs are saved in their respective directories.

### 6. Launch the Streamlit Dashboard

After the pipeline completes successfully:

```bash
python -m streamlit run app.py
```

The dashboard will open in your browser.

---

## Key Files

| File | Description |
|---|---|
| `pipeline.py` | End-to-end data processing and modeling pipeline |
| `app.py` | Interactive Streamlit dashboard |
| `requirements.txt` | Project dependencies |
| `notebooks/` | Exploratory analysis and project notebooks |
| `sql/queries.sql` | SQL business analysis queries |
| `models/` | Saved XGBoost model and metadata |
| `outputs/metrics.json` | Consolidated project metrics |

---

## Skills Demonstrated

- Data Cleaning & Preprocessing
- Exploratory Data Analysis
- Data Integration & Feature Engineering
- SQL & Relational Data Analysis
- Statistical Hypothesis Testing
- Classification & Model Evaluation
- Imbalanced Classification
- Explainable AI (SHAP)
- Unsupervised Learning & Customer Segmentation
- Natural Language Processing
- Interactive Data Visualization
- End-to-End Data Science Workflow

---

## Future Improvements

- Experiment with advanced NLP models and transformer-based sentiment classification.
- Improve late-delivery prediction through feature selection and hyperparameter optimization.
- Explore customer lifetime value (CLV) prediction.
- Deploy the Streamlit dashboard to a public hosting platform.
- Add automated model monitoring and data validation.

---

## Dataset & Acknowledgment

This project uses the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), available on Kaggle.

The dataset contains anonymized e-commerce order information and related customer, seller, product, payment, and review records.

---

## Author

**Hulash Kumar Shah**

B.Sc. CSIT Student | Data Science & Machine Learning

- GitHub: [@Hulash-shah](https://github.com/Hulash-shah)
- Project: [Olist E-Commerce Analytics & Predictive Modeling](https://github.com/Hulash-shah/Olist-E-commerce-Analysis)

---

*Built as a practical Data Science project to explore real-world e-commerce data and develop analytical and predictive solutions.*
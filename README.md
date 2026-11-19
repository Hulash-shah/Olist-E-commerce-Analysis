# Olist E-Commerce Analytics and Predictive Modeling

End-to-end data science project on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (~100k orders, 2016–2018): data integration, SQL, EDA, statistics, late-delivery prediction, customer segmentation, review NLP, SHAP explainability and a Streamlit dashboard.

## Repository layout
| Path | Content |
|---|---|
| `pipeline.py` | Reproducible pipeline (integration → SQL → EDA → stats → ML → K-Means → NLP → SHAP) |
| `notebooks/brazilian-e-commerce_EDA.ipynb` | Detailed data audit, EDA, cohort retention and RFM analysis |
| `app.py` | Streamlit dashboard (4 pages, includes live order scoring and sentiment tester) |
| `sql/queries.sql` | SQL analyses (run via SQLite inside the pipeline) |
| `data/` | Cleaned master table, delivered-orders modeling set, segments, sentiment scores (gzipped) |
| `outputs/` | `metrics.json`, model comparison, segment profiles, SQL results, 12 figures |
| `models/` | Trained XGBoost late-delivery model and sentiment model |

## Run
```bash
pip install -r requirements.txt
# put the 9 Olist CSVs somewhere and set RAW in pipeline.py, then:
python pipeline.py
streamlit run app.py
```

## Method
- **Integration**: items, payments and reviews are aggregated to one row per order *before* joining (no row duplication). Geolocation is reduced to one centroid per zip prefix; seller–customer distance is computed with the haversine formula. Categories are translated to English.
- **Target**: `is_late = 1` if `order_delivered_customer_date > order_estimated_delivery_date` (delivered orders only, 96,427 orders, 8.1% late).
- **Leakage control**: only information available at purchase/approval is used (price, freight, weight, distance, estimated delivery days, states, category, payment info, purchase timing). Actual delivery timestamps are never features.
- **Models**: Logistic Regression (scaled), Random Forest, XGBoost, all class-weighted, 80/20 stratified split. Because only ~8% of orders are late, **PR-AUC** is the headline metric; thresholds are chosen to maximise F1.

## Results
| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.714 | 0.185 | 0.187 | 0.437 | 0.262 |
| Random Forest | 0.794 | 0.295 | 0.300 | 0.468 | 0.365 |
| **XGBoost** | **0.799** | **0.306** | 0.311 | 0.470 | 0.374 |

The test-set base rate is ~8%, so XGBoost's PR-AUC of 0.31 is roughly 4x a random ranking, useful for flagging risky orders, but not a precise predictor.

**Statistics**
- Late orders average **2.56** stars vs **4.29** for on-time (Mann–Whitney p < 0.001, rank-biserial effect 0.55, a large effect).
- Same-state orders are late **6.1%** of the time vs **9.2%** cross-state (chi-square p < 0.001).
- Distance and delivery time: Spearman ρ = 0.54.

**SHAP drivers** (mean |SHAP|): purchase month, estimated delivery days, seller–customer distance, customer state (SP/MG/RJ), approval time, same-state flag, freight.

**Segmentation** (K-Means, k=4, features: log recency/frequency/monetary/freight ratio + review score)
- *Satisfied high spenders* 51% (avg R$212, review 4.7), *Low-value one-time* 28% (R$54, high freight burden), *Dissatisfied* 18% (review 1.7), *Repeat buyers* 3%.
- Only ~3% of customers ever buy twice, retention is the main commercial gap.

**Sentiment (NLP)**: TF-IDF + Logistic Regression on 35k Portuguese review comments (score ≥4 positive, ≤2 negative, 3 dropped): ROC-AUC **0.975**, macro-F1 **0.916**. Top negative terms: *não recebi, péssimo, demora, quebrado, diferente*. **74.7%** of comments on late orders are negative vs **20.1%** on time.

## Limitations
- Random split, not time-based; `purchase_month` is the top SHAP feature because late spikes cluster in Nov 2017 and Feb–Mar 2018 (seasonal carrier congestion), so performance on a future period would likely be lower.
- Sentiment labels are derived from star ratings, not human annotation; the model is TF-IDF-based and does not handle sarcasm.
- Forecasting, recommendation and anomaly detection were intentionally left out (the proposal makes them optional "where supported by the data").
- Dataset is anonymised and covers only 2016–2018.

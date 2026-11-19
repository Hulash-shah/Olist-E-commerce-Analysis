"""
Olist E-Commerce Analytics: end-to-end pipeline
================================================

Stages
    1. Load and integrate the nine Olist tables into one row per order
    2. SQL analysis (SQLite)
    3. Exploratory data analysis (figures)
    4. Statistical tests
    5. Late-delivery prediction (Logistic Regression, Random Forest, XGBoost)
    6. SHAP explainability
    7. Customer segmentation (K-Means)
    8. Review sentiment analysis (NLP)

Usage
    python pipeline.py --raw path/to/olist_csv_folder
"""
import argparse
import json
import sqlite3
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve,
                             silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ======================================================================
# CONFIGURATION
# ======================================================================
BASE = Path(__file__).parent
DATA_DIR, OUT_DIR, MODEL_DIR, SQL_DIR = BASE / "data", BASE / "outputs", BASE / "models", BASE / "sql"
FIG_DIR = OUT_DIR / "figures"

SEED = 42
TEST_SIZE = 0.2
MAX_DELIVERY_DAYS = 120      # drop impossible delivery times
TOP_CATEGORIES = 25          # other categories are grouped as "other"
N_CLUSTERS = 4
SHAP_SAMPLE = 2000
BLUE, RED, GREEN = "#2a6f97", "#c1121f", "#2a9d8f"

# Features used by the late-delivery model. All are known at purchase/approval
# time; the actual delivery dates are never used (no data leakage).
NUMERIC_FEATURES = ["n_items", "n_sellers", "price", "freight", "freight_ratio", "weight_g",
                    "volume_cm3", "photos", "distance_km", "same_state", "estimated_days",
                    "purchase_dow", "purchase_hour", "purchase_m", "approval_hours",
                    "installments", "payment_value"]
CATEGORICAL_FEATURES = ["customer_state", "seller_state", "payment_type", "category"]

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110


def save_fig(name: str) -> None:
    """Save the current matplotlib figure to outputs/figures and close it."""
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close()


# ======================================================================
# 1. LOAD AND INTEGRATE
# ======================================================================
def load_tables(raw: Path) -> dict:
    """Read the nine Olist CSV files into a dict of DataFrames."""
    files = {
        "customers": "olist_customers_dataset.csv", "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv", "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv", "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv", "geo": "olist_geolocation_dataset.csv",
        "translation": "product_category_name_translation.csv",
    }
    tables = {name: pd.read_csv(raw / file) for name, file in files.items()}
    for col in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                "order_delivered_customer_date", "order_estimated_delivery_date"]:
        tables["orders"][col] = pd.to_datetime(tables["orders"][col], errors="coerce")
    return tables


def prepare_geolocation(geo: pd.DataFrame) -> pd.DataFrame:
    """Reduce ~1M rows to one average coordinate per zip prefix (Brazil only)."""
    inside_brazil = geo.geolocation_lat.between(-34, 6) & geo.geolocation_lng.between(-74, -34)
    return geo[inside_brazil].groupby("geolocation_zip_code_prefix")[
        ["geolocation_lat", "geolocation_lng"]].mean()


def prepare_products(products: pd.DataFrame, translation: pd.DataFrame) -> pd.DataFrame:
    """Translate categories to English and fill missing size/weight values with the median."""
    products = products.merge(translation, on="product_category_name", how="left")
    products["category"] = products.product_category_name_english.fillna("unknown")
    products["volume_cm3"] = (products.product_length_cm * products.product_height_cm
                              * products.product_width_cm)
    for col in ["product_weight_g", "volume_cm3", "product_photos_qty"]:
        products[col] = products[col].fillna(products[col].median())
    return products


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    """Great-circle distance in kilometres between two points."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def build_master(t: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the order-level master table (exactly one row per order) and the
    modelling set of delivered orders with the `is_late` target.
    Items, payments and reviews are aggregated BEFORE joining so no rows duplicate.
    """
    products = prepare_products(t["products"], t["translation"])
    geo = prepare_geolocation(t["geo"])

    # items -> one row per order
    items = t["items"].merge(
        products[["product_id", "category", "product_weight_g", "volume_cm3", "product_photos_qty"]],
        on="product_id", how="left")
    items_o = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"), n_sellers=("seller_id", "nunique"),
        price=("price", "sum"), freight=("freight_value", "sum"),
        weight_g=("product_weight_g", "sum"), volume_cm3=("volume_cm3", "sum"),
        photos=("product_photos_qty", "mean"),
        category=("category", lambda s: s.mode().iat[0]), seller_id=("seller_id", "first"),
    ).reset_index().merge(
        t["sellers"][["seller_id", "seller_zip_code_prefix", "seller_state"]], on="seller_id", how="left")

    # payments and reviews -> one row per order
    payments_o = t["payments"].groupby("order_id").agg(
        payment_value=("payment_value", "sum"), installments=("payment_installments", "max"),
        payment_type=("payment_type", lambda s: s.mode().iat[0])).reset_index()
    reviews_o = t["reviews"].sort_values("review_creation_date").groupby("order_id").agg(
        review_score=("review_score", "last"), review_comment=("review_comment_message", "last")
    ).reset_index()

    df = (t["orders"].merge(t["customers"], on="customer_id", how="left")
          .merge(items_o, on="order_id", how="inner")
          .merge(payments_o, on="order_id", how="left")
          .merge(reviews_o, on="order_id", how="left")
          .merge(geo.add_prefix("c_"), left_on="customer_zip_code_prefix", right_index=True, how="left")
          .merge(geo.add_prefix("s_"), left_on="seller_zip_code_prefix", right_index=True, how="left"))

    # derived columns
    days = lambda end, start: (df[end] - df[start]).dt.total_seconds() / 86400
    df["distance_km"] = haversine_km(df.c_geolocation_lat, df.c_geolocation_lng,
                                     df.s_geolocation_lat, df.s_geolocation_lng)
    df["distance_km"] = df.distance_km.fillna(df.distance_km.median())
    df["same_state"] = (df.customer_state == df.seller_state).astype(int)
    df["delivery_days"] = days("order_delivered_customer_date", "order_purchase_timestamp")
    df["estimated_days"] = days("order_estimated_delivery_date", "order_purchase_timestamp")
    df["delay_days"] = days("order_delivered_customer_date", "order_estimated_delivery_date")
    ts = df.order_purchase_timestamp
    df["purchase_month"] = ts.dt.to_period("M").astype(str)
    df["purchase_dow"], df["purchase_hour"], df["purchase_m"] = ts.dt.dayofweek, ts.dt.hour, ts.dt.month
    df["approval_hours"] = ((df.order_approved_at - ts).dt.total_seconds() / 3600).clip(0, 200)
    df["freight_ratio"] = df.freight / (df.price + 1e-6)
    df["payment_value"] = df.payment_value.fillna(df.price + df.freight)
    df["installments"] = df.installments.fillna(1)
    df["payment_type"] = df.payment_type.fillna("unknown")

    # modelling set: delivered orders; late = delivered AFTER the estimated date
    delivered = df[(df.order_status == "delivered") & df.delivery_days.notna()].copy()
    delivered["is_late"] = (delivered.delay_days > 0).astype(int)
    delivered = delivered[delivered.delivery_days.between(0, MAX_DELIVERY_DAYS)]
    return df, delivered


# ======================================================================
# 2. SQL ANALYSIS
# ======================================================================
SQL_QUERIES = {
    "late_rate_by_state": """
        SELECT customer_state, COUNT(*) n, ROUND(100.0*AVG(is_late),1) late_pct,
               ROUND(AVG(delivery_days),1) avg_days
        FROM orders GROUP BY customer_state HAVING n>=300 ORDER BY late_pct DESC LIMIT 10""",
    "revenue_by_category": """
        SELECT category, COUNT(*) orders, ROUND(SUM(price)) revenue, ROUND(AVG(review_score),2) avg_review
        FROM orders GROUP BY category ORDER BY revenue DESC LIMIT 10""",
    "monthly_trend": """
        SELECT purchase_month, COUNT(*) orders, ROUND(SUM(payment_value)) revenue,
               ROUND(100.0*AVG(is_late),1) late_pct
        FROM orders WHERE purchase_month BETWEEN '2017-01' AND '2018-08'
        GROUP BY purchase_month ORDER BY purchase_month""",
    "review_by_lateness": """
        SELECT is_late, COUNT(*) n, ROUND(AVG(review_score),2) avg_review,
               ROUND(100.0*AVG(review_score<=2),1) pct_negative
        FROM orders GROUP BY is_late""",
}


def run_sql_analysis(delivered: pd.DataFrame, metrics: dict) -> dict:
    """Load delivered orders into in-memory SQLite, run the queries, save results."""
    con = sqlite3.connect(":memory:")
    delivered.drop(columns=["order_approved_at"]).to_sql("orders", con, index=False)
    (SQL_DIR / "queries.sql").write_text(
        "\n\n".join(f"-- {name}\n{query.strip()};" for name, query in SQL_QUERIES.items()))
    results = {name: pd.read_sql(query, con) for name, query in SQL_QUERIES.items()}
    for name, frame in results.items():
        frame.to_csv(OUT_DIR / f"sql_{name}.csv", index=False)
    metrics["sql_review_by_lateness"] = results["review_by_lateness"].to_dict("records")
    return results


# ======================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ======================================================================
def make_eda_figures(delivered: pd.DataFrame, sql: dict) -> None:
    """Draw the six exploratory charts."""
    monthly = sql["monthly_trend"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(monthly.purchase_month, monthly.revenue / 1e3, marker="o", color=BLUE)
    ax[0].set_title("Monthly revenue (R$ thousands)")
    ax[1].bar(monthly.purchase_month, monthly.late_pct, color=RED)
    ax[1].set_title("Late delivery rate by month (%)")
    for a in ax:
        a.tick_params(axis="x", rotation=70)
    save_fig("01_monthly_trend")

    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=sql["revenue_by_category"], y="category", x="revenue", color=BLUE)
    plt.title("Top 10 categories by revenue (R$)")
    save_fig("02_top_categories")

    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=sql["late_rate_by_state"], y="customer_state", x="late_pct", color=RED)
    plt.title("Late delivery % by customer state (top 10, n>=300)")
    save_fig("03_late_by_state")

    plt.figure(figsize=(8, 4))
    sns.histplot(delivered.delivery_days.clip(upper=60), bins=60, color=BLUE)
    plt.axvline(delivered.delivery_days.median(), color="k", ls="--")
    plt.title(f"Delivery time (median {delivered.delivery_days.median():.1f} days)")
    save_fig("04_delivery_time")

    plt.figure(figsize=(6, 4))
    sns.barplot(data=delivered, x="is_late", y="review_score", errorbar="ci")
    plt.xticks([0, 1], ["On time", "Late"])
    plt.title("Average review score vs lateness")
    save_fig("05_review_vs_late")

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=delivered, x="same_state", y="delivery_days", showfliers=False)
    plt.xticks([0, 1], ["Different state", "Same state"])
    plt.title("Delivery days: seller and customer in the same state?")
    save_fig("06_same_state")


# ======================================================================
# 4. STATISTICAL ANALYSIS
# ======================================================================
def run_statistics(delivered: pd.DataFrame, metrics: dict) -> None:
    """Mann-Whitney (reviews vs lateness), chi-square (same state vs late), Spearman (distance)."""
    rated = delivered.dropna(subset=["review_score"])
    late, ontime = rated[rated.is_late == 1].review_score, rated[rated.is_late == 0].review_score
    mw = stats.mannwhitneyu(late, ontime)
    rank_biserial = 1 - 2 * mw.statistic / (len(late) * len(ontime))

    table = pd.crosstab(delivered.same_state, delivered.is_late)
    chi = stats.chi2_contingency(table)
    rho = stats.spearmanr(delivered.distance_km, delivered.delivery_days)

    metrics["stats"] = {
        "mannwhitney_review_late_vs_ontime": dict(
            p=float(mw.pvalue), rank_biserial=round(float(rank_biserial), 3),
            mean_late=round(late.mean(), 2), mean_ontime=round(ontime.mean(), 2)),
        "chi2_same_state_vs_late": dict(
            chi2=round(chi[0], 1), p=float(chi[1]),
            late_pct_same=round(table.loc[1, 1] / table.loc[1].sum() * 100, 2),
            late_pct_diff=round(table.loc[0, 1] / table.loc[0].sum() * 100, 2)),
        "spearman_distance_vs_days": dict(rho=round(float(rho.correlation), 3), p=float(rho.pvalue)),
    }
    print(json.dumps(metrics["stats"], indent=2))


# ======================================================================
# 5. LATE-DELIVERY MODELS
# ======================================================================
def build_feature_matrix(delivered: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Index]:
    """One-hot encode categoricals (rare categories -> 'other') and impute numerics."""
    top_cats = delivered.category.value_counts().head(TOP_CATEGORIES).index
    X = delivered[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    X["category"] = X.category.where(X.category.isin(top_cats), "other")
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(X[NUMERIC_FEATURES].median())
    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=int)
    return X, delivered.is_late, top_cats


def best_f1_threshold(y_true, proba) -> float:
    """Probability threshold that maximises F1 on the precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return thresholds[np.argmax(f1[:-1])]


def train_late_models(delivered: pd.DataFrame, metrics: dict):
    """Train and compare three models. Returns (xgb_model, X_test, y_test, proba_dict, X_columns, top_cats)."""
    X, y, top_cats = build_feature_matrix(delivered)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    scaler = StandardScaler().fit(X_tr[NUMERIC_FEATURES])          # scaling only for Logistic Regression
    X_tr_s, X_te_s = X_tr.copy(), X_te.copy()
    X_tr_s[NUMERIC_FEATURES] = scaler.transform(X_tr[NUMERIC_FEATURES])
    X_te_s[NUMERIC_FEATURES] = scaler.transform(X_te[NUMERIC_FEATURES])

    imbalance = (y_tr == 0).sum() / (y_tr == 1).sum()
    candidates = {   # name: (model, needs_scaled_features)
        "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5), True),
        "Random Forest": (RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                          class_weight="balanced_subsample", n_jobs=-1, random_state=SEED), False),
        "XGBoost": (XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, scale_pos_weight=imbalance, eval_metric="aucpr",
                    n_jobs=-1, random_state=SEED), False),
    }

    rows, probas = [], {}
    for name, (model, scaled) in candidates.items():
        model.fit(X_tr_s if scaled else X_tr, y_tr)
        proba = model.predict_proba(X_te_s if scaled else X_te)[:, 1]
        threshold = best_f1_threshold(y_te, proba)
        pred = (proba >= threshold).astype(int)
        probas[name] = proba
        rows.append(dict(model=name, roc_auc=roc_auc_score(y_te, proba),
                         pr_auc=average_precision_score(y_te, proba), threshold=threshold,
                         precision=precision_score(y_te, pred), recall=recall_score(y_te, pred),
                         f1=f1_score(y_te, pred)))
    results = pd.DataFrame(rows).round(4)
    results.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print(results)
    metrics["models"] = results.to_dict("records")
    metrics["baseline_late_rate_test"] = round(float(y_te.mean()), 4)
    metrics["best_model"] = results.sort_values("pr_auc").iloc[-1].model

    # ROC and precision-recall curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_te, proba)
        ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_te, proba):.3f})")
        prec, rec, _ = precision_recall_curve(y_te, proba)
        ax[1].plot(rec, prec, label=f"{name} ({average_precision_score(y_te, proba):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_title("ROC"); ax[0].legend()
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[1].axhline(y_te.mean(), color="k", ls="--"); ax[1].set_title("Precision-Recall"); ax[1].legend()
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    save_fig("07_model_curves")

    # save the XGBoost model in its portable JSON format
    xgb = candidates["XGBoost"][0]
    xgb.save_model(str(MODEL_DIR / "xgb_late.json"))
    json.dump(dict(columns=list(X.columns), top_cats=list(top_cats), num=NUMERIC_FEATURES),
              open(MODEL_DIR / "xgb_meta.json", "w"))
    return xgb, X_te, y_te, probas["XGBoost"]


# ======================================================================
# 6. SHAP EXPLAINABILITY
# ======================================================================
def explain_with_shap(xgb, X_test: pd.DataFrame, metrics: dict) -> None:
    """SHAP summary and bar plots on a sample of test orders."""
    sample = X_test.sample(SHAP_SAMPLE, random_state=SEED)
    values = shap.TreeExplainer(xgb).shap_values(sample)
    shap.summary_plot(values, sample, max_display=15, show=False)
    save_fig("08_shap_summary")
    shap.summary_plot(values, sample, plot_type="bar", max_display=15, show=False)
    save_fig("09_shap_bar")
    importance = pd.Series(np.abs(values).mean(0), index=sample.columns).sort_values(ascending=False).head(10)
    importance.to_csv(OUT_DIR / "shap_top_features.csv")
    metrics["shap_top"] = importance.round(3).to_dict()


# ======================================================================
# 7. CUSTOMER SEGMENTATION
# ======================================================================
def name_segments(profile: pd.DataFrame) -> pd.Series:
    """Give each cluster a business-friendly name based on its average behaviour."""
    median_spend = profile.monetary.median()

    def label(row):
        if row.frequency > 1.3:
            return "Repeat buyers"
        if row.avg_review < 3:
            return "Dissatisfied"
        return "High-value one-time" if row.monetary > median_spend else "Low-value one-time"

    names = profile.apply(label, axis=1)
    for name, group in profile.groupby(names):            # disambiguate duplicate names by recency
        if len(group) > 1:
            for i, idx in enumerate(group.sort_values("recency").index):
                names[idx] = f"{name} ({'recent' if i == 0 else 'lapsed'})"
    return names


def segment_customers(df: pd.DataFrame, metrics: dict) -> None:
    """K-Means on log-scaled RFM + freight burden + review score."""
    snapshot = df.order_purchase_timestamp.max() + pd.Timedelta(days=1)
    cust = df[df.order_status != "canceled"].groupby("customer_unique_id").agg(
        recency=("order_purchase_timestamp", lambda s: (snapshot - s.max()).days),
        frequency=("order_id", "nunique"), monetary=("payment_value", "sum"),
        avg_review=("review_score", "mean"), avg_freight_ratio=("freight_ratio", "mean")).reset_index()
    cust["avg_review"] = cust.avg_review.fillna(cust.avg_review.median())

    features = np.log1p(cust[["recency", "frequency", "monetary", "avg_freight_ratio"]]).join(cust[["avg_review"]])
    Z = StandardScaler().fit_transform(features)

    # choose k with silhouette scores on a 15,000-customer sample
    sample = np.random.RandomState(SEED).choice(len(Z), 15000, replace=False)
    silhouettes = {k: silhouette_score(Z[sample], KMeans(k, n_init=5, random_state=SEED).fit_predict(Z[sample]))
                   for k in range(2, 8)}
    plt.figure(figsize=(5, 3.5))
    plt.plot(list(silhouettes), list(silhouettes.values()), marker="o")
    plt.title("Silhouette score by k"); plt.xlabel("k")
    save_fig("10_silhouette")

    cust["cluster"] = KMeans(N_CLUSTERS, n_init=10, random_state=SEED).fit_predict(Z)
    profile = cust.groupby("cluster").agg(
        customers=("customer_unique_id", "count"), recency=("recency", "mean"),
        frequency=("frequency", "mean"), monetary=("monetary", "mean"),
        avg_review=("avg_review", "mean"), freight_ratio=("avg_freight_ratio", "mean")).round(2)
    profile["segment"] = name_segments(profile)
    profile["share_pct"] = (profile.customers / profile.customers.sum() * 100).round(1)
    cust["segment"] = cust.cluster.map(profile.segment)

    profile.to_csv(OUT_DIR / "segment_profiles.csv")
    cust.to_csv(DATA_DIR / "customer_segments.csv", index=False)
    metrics["segmentation"] = dict(k=N_CLUSTERS, silhouette_k=round(silhouettes[N_CLUSTERS], 3),
                                   profiles=profile.reset_index().to_dict("records"))
    print(profile)

    plt.figure(figsize=(7, 4.5))
    sns.scatterplot(data=cust.sample(8000, random_state=1), x="recency", y="monetary",
                    hue="segment", s=12, alpha=0.6)
    plt.yscale("log"); plt.title("Customer segments (K-Means)")
    save_fig("11_segments")


# ======================================================================
# 8. NLP SENTIMENT ANALYSIS
# ======================================================================
def analyze_sentiment(delivered: pd.DataFrame, metrics: dict) -> None:
    """TF-IDF + Logistic Regression on review comments (4-5 stars positive, 1-2 negative)."""
    reviews = delivered.dropna(subset=["review_comment"]).copy()
    reviews = reviews[reviews.review_score != 3]
    reviews["label"] = (reviews.review_score >= 4).astype(int)

    text_tr, text_te, y_tr, y_te = train_test_split(
        reviews.review_comment, reviews.label, test_size=TEST_SIZE, stratify=reviews.label, random_state=SEED)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=30000,
                                 strip_accents="unicode", lowercase=True)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=2)
    clf.fit(vectorizer.fit_transform(text_tr), y_tr)
    proba = clf.predict_proba(vectorizer.transform(text_te))[:, 1]

    # most negative / positive terms
    coefs = pd.Series(clf.coef_[0], index=vectorizer.get_feature_names_out()).sort_values()
    negative, positive = coefs.head(15), coefs.tail(15)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    negative.iloc[::-1].plot.barh(ax=ax[0], color=RED); ax[0].set_title("Terms pointing to negative")
    positive.plot.barh(ax=ax[1], color=GREEN); ax[1].set_title("Terms pointing to positive")
    save_fig("12_sentiment_terms")

    reviews["sentiment_pred"] = clf.predict_proba(vectorizer.transform(reviews.review_comment))[:, 1]
    reviews[["order_id", "review_score", "review_comment", "sentiment_pred", "is_late"]].to_csv(
        DATA_DIR / "reviews_sentiment.csv.gz", index=False)

    metrics["nlp"] = dict(
        n_reviews=len(reviews), pct_positive=round(reviews.label.mean() * 100, 1),
        roc_auc=round(roc_auc_score(y_te, proba), 4),
        f1_macro=round(f1_score(y_te, (proba > 0.5).astype(int), average="macro"), 4),
        top_negative_terms=list(negative.index[:10]), top_positive_terms=list(positive.index[::-1][:10]),
        late_vs_neg_share=dict(
            late_neg_pct=round((reviews[reviews.is_late == 1].label == 0).mean() * 100, 1),
            ontime_neg_pct=round((reviews[reviews.is_late == 0].label == 0).mean() * 100, 1)))
    print(metrics["nlp"])


# ======================================================================
# MAIN
# ======================================================================
def main(raw: Path) -> None:
    for folder in (DATA_DIR, FIG_DIR, MODEL_DIR, SQL_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    metrics = {}

    print("1/8 Loading and integrating data...")
    tables = load_tables(raw)
    master, delivered = build_master(tables)
    master.to_csv(DATA_DIR / "olist_master.csv.gz", index=False)
    delivered.to_csv(DATA_DIR / "olist_delivered.csv.gz", index=False)
    metrics["data"] = dict(orders_raw=len(tables["orders"]), orders_master=len(master),
                           delivered_modeled=len(delivered), late_rate=round(delivered.is_late.mean(), 4))
    print(metrics["data"])

    print("2/8 SQL analysis...");              sql = run_sql_analysis(delivered, metrics)
    print("3/8 Exploratory analysis...");      make_eda_figures(delivered, sql)
    print("4/8 Statistical tests...");         run_statistics(delivered, metrics)
    print("5/8 Late-delivery models...");      xgb, X_test, y_test, xgb_proba = train_late_models(delivered, metrics)
    print("6/8 SHAP explainability...");       explain_with_shap(xgb, X_test, metrics)
    print("7/8 Customer segmentation...");     segment_customers(master, metrics)
    print("8/8 Sentiment analysis...");        analyze_sentiment(delivered, metrics)

    json.dump(metrics, open(OUT_DIR / "metrics.json", "w"), indent=2, default=float)
    print("Done. Results are in the 'outputs', 'data' and 'models' folders.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Olist end-to-end pipeline.")
    parser.add_argument("--raw", type=Path, default=Path("raw"), help="folder containing the 9 Olist CSV files")
    main(parser.parse_args().raw)
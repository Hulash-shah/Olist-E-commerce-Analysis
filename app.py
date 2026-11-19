"""
Olist E-Commerce Analytics: Streamlit dashboard
===============================================

Pages
    Overview             KPIs, trends, late rate by state, top categories (with filters)
    Late-delivery model  model comparison, curves, SHAP and a live order scorer
    Customer segments    K-Means segment profiles
    Review sentiment     NLP metrics and a live sentiment tester

Run with:  streamlit run app.py
(Run pipeline.py first if the data/, models/ and outputs/ folders are missing.)
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ======================================================================
# CONFIGURATION
# ======================================================================
BASE = Path(__file__).parent
OUT_DIR, DATA_DIR, MODEL_DIR = BASE / "outputs", BASE / "data", BASE / "models"
FIG_DIR = OUT_DIR / "figures"

MIN_STATE_ORDERS = 300                     # states with fewer orders are not ranked
TREND_START, TREND_END = "2017-01", "2018-08"
PAGES = ["Overview", "Late-delivery model", "Customer segments", "Review sentiment"]

ORDER_COLUMNS = ["order_id", "order_purchase_timestamp", "purchase_month", "customer_state",
                 "seller_state", "category", "price", "freight", "payment_value",
                 "delivery_days", "estimated_days", "is_late", "review_score", "distance_km"]

st.set_page_config(page_title="Olist Analytics", layout="wide")


# ======================================================================
# LOADING (cached so each file is read only once)
# ======================================================================
@st.cache_data
def load_orders() -> pd.DataFrame:
    """Delivered orders produced by pipeline.py."""
    return pd.read_csv(DATA_DIR / "olist_delivered.csv.gz",
                       parse_dates=["order_purchase_timestamp"], usecols=ORDER_COLUMNS)


@st.cache_resource
def load_models():
    """
    Return (late-delivery model bundle, sentiment model).
    XGBoost is loaded from its portable JSON file. The sentiment model is
    trained at start-up from the saved review data (takes a few seconds),
    which avoids version problems with pickled models.
    """
    xgb = XGBClassifier()
    xgb.load_model(str(MODEL_DIR / "xgb_late.json"))
    late_bundle = dict(model=xgb, **json.load(open(MODEL_DIR / "xgb_meta.json")))

    reviews = pd.read_csv(DATA_DIR / "reviews_sentiment.csv.gz")
    reviews = reviews[reviews.review_score != 3]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=30000, strip_accents="unicode")
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", C=2)
    classifier.fit(vectorizer.fit_transform(reviews.review_comment), (reviews.review_score >= 4).astype(int))
    return late_bundle, dict(vectorizer=vectorizer, classifier=classifier)


# ======================================================================
# PAGE 1: OVERVIEW
# ======================================================================
def filter_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Sidebar filters for customer state and product category."""
    states = st.sidebar.multiselect("Customer state", sorted(orders.customer_state.unique()))
    categories = st.sidebar.multiselect("Category", sorted(orders.category.unique()))
    mask = pd.Series(True, index=orders.index)
    if states:
        mask &= orders.customer_state.isin(states)
    if categories:
        mask &= orders.category.isin(categories)
    return orders[mask]


def page_overview(orders: pd.DataFrame, metrics: dict) -> None:
    st.title("Olist E-Commerce Overview")
    d = filter_orders(orders)

    # KPI row
    kpi = st.columns(4)
    kpi[0].metric("Delivered orders", f"{len(d):,}")
    kpi[1].metric("Revenue (R$)", f"{d.payment_value.sum():,.0f}")
    kpi[2].metric("Late delivery rate", f"{d.is_late.mean() * 100:.1f}%")
    kpi[3].metric("Avg review", f"{d.review_score.mean():.2f}")

    # monthly revenue and late rate
    monthly = (d[d.purchase_month.between(TREND_START, TREND_END)]
               .groupby("purchase_month")
               .agg(revenue=("payment_value", "sum"), late_pct=("is_late", lambda s: s.mean() * 100))
               .reset_index())
    left, right = st.columns(2)
    left.plotly_chart(px.line(monthly, x="purchase_month", y="revenue",
                              title="Monthly revenue (R$)", markers=True), width="stretch")
    right.plotly_chart(px.bar(monthly, x="purchase_month", y="late_pct",
                              title="Late delivery % by month"), width="stretch")

    # late rate by state and top categories
    by_state = (d.groupby("customer_state")
                .agg(n=("order_id", "count"), late_pct=("is_late", lambda s: s.mean() * 100))
                .reset_index())
    by_state = by_state[by_state.n >= MIN_STATE_ORDERS].sort_values("late_pct", ascending=False)
    top_categories = d.groupby("category").payment_value.sum().nlargest(10).reset_index()
    left, right = st.columns(2)
    left.plotly_chart(px.bar(by_state, x="customer_state", y="late_pct",
                             title=f"Late % by state (n≥{MIN_STATE_ORDERS})"), width="stretch")
    right.plotly_chart(px.bar(top_categories, y="category", x="payment_value", orientation="h",
                              title="Top categories by revenue"), width="stretch")

    review = metrics["stats"]["mannwhitney_review_late_vs_ontime"]
    st.info(f"Late orders average **{review['mean_late']}** stars "
            f"vs **{review['mean_ontime']}** for on-time orders.")


# ======================================================================
# PAGE 2: LATE-DELIVERY MODEL
# ======================================================================
def build_order_row(bundle: dict, inputs: dict) -> pd.DataFrame:
    """
    Turn the user's form inputs into one model-ready row.
    Fields the form does not ask about use typical defaults.
    """
    row = pd.DataFrame(0, index=[0], columns=bundle["columns"])
    numeric = dict(
        n_items=1, n_sellers=1, price=inputs["price"], freight=inputs["freight"],
        freight_ratio=inputs["freight"] / inputs["price"], weight_g=1000, volume_cm3=10000, photos=2,
        distance_km=inputs["distance"], same_state=int(inputs["customer_state"] == inputs["seller_state"]),
        estimated_days=inputs["estimated_days"], purchase_dow=2, purchase_hour=14,
        purchase_m=inputs["month"], approval_hours=inputs["approval_hours"],
        installments=1, payment_value=inputs["price"] + inputs["freight"])
    for column, value in numeric.items():
        row[column] = value

    category = inputs["category"] if inputs["category"] in bundle["top_cats"] else "other"
    one_hot = [f"customer_state_{inputs['customer_state']}", f"seller_state_{inputs['seller_state']}",
               f"category_{category}", "payment_type_credit_card"]
    for column in one_hot:
        if column in row:                      # the reference level has no column
            row[column] = 1
    return row


def order_form(orders: pd.DataFrame) -> dict:
    """Input widgets for the order scorer. Returns the chosen values."""
    customer_states = sorted(orders.customer_state.unique())
    seller_states = sorted(orders.seller_state.dropna().unique())
    col = st.columns(3)
    return dict(
        customer_state=col[0].selectbox("Customer state", customer_states, index=customer_states.index("SP")),
        seller_state=col[1].selectbox("Seller state", seller_states, index=seller_states.index("SP")),
        category=col[2].selectbox("Category", sorted(orders.category.unique())),
        price=col[0].number_input("Price (R$)", 1.0, 5000.0, 100.0),
        freight=col[1].number_input("Freight (R$)", 0.0, 500.0, 20.0),
        distance=col[2].slider("Seller–customer distance (km)", 0, 3500, 500),
        estimated_days=col[0].slider("Estimated delivery (days)", 3, 60, 24),
        month=col[1].slider("Purchase month", 1, 12, 6),
        approval_hours=col[2].slider("Approval time (hours)", 0, 100, 2),
    )


def page_late_model(orders: pd.DataFrame, metrics: dict, bundle: dict) -> None:
    st.title("Predict late deliveries")
    st.dataframe(pd.read_csv(OUT_DIR / "model_comparison.csv"), hide_index=True)
    st.caption(f"Baseline late rate in test set: {metrics['baseline_late_rate_test'] * 100:.1f}%. "
               "PR-AUC is the headline metric because the classes are imbalanced.")
    left, right = st.columns(2)
    left.image(str(FIG_DIR / "07_model_curves.png"))
    right.image(str(FIG_DIR / "08_shap_summary.png"))

    st.subheader("Score an order (XGBoost)")
    row = build_order_row(bundle, order_form(orders))
    probability = float(bundle["model"].predict_proba(row)[0, 1])
    difference = (probability - metrics["data"]["late_rate"]) * 100
    st.metric("Late-delivery probability", f"{probability * 100:.1f}%",
              delta=f"{difference:+.1f} pts vs average", delta_color="inverse")


# ======================================================================
# PAGE 3: CUSTOMER SEGMENTS
# ======================================================================
def page_segments(metrics: dict) -> None:
    st.title(f"Customer segments (K-Means, k={metrics['segmentation']['k']})")
    profiles = pd.read_csv(OUT_DIR / "segment_profiles.csv")
    st.dataframe(profiles, hide_index=True)

    left, right = st.columns(2)
    left.plotly_chart(px.pie(profiles, names="segment", values="customers",
                             title="Segment share"), width="stretch")
    right.plotly_chart(px.bar(profiles, x="segment", y="monetary", color="avg_review",
                              title="Avg spend (R$), coloured by avg review"), width="stretch")
    st.caption(f"Silhouette score: {metrics['segmentation']['silhouette_k']}. Segments are driven by spend, "
               "review score and freight burden; only ~3% of customers ever repurchase.")


# ======================================================================
# PAGE 4: REVIEW SENTIMENT
# ======================================================================
def page_sentiment(metrics: dict, sentiment: dict) -> None:
    nlp = metrics["nlp"]
    st.title("Review sentiment (NLP)")
    kpi = st.columns(3)
    kpi[0].metric("Reviews with text", f"{nlp['n_reviews']:,}")
    kpi[1].metric("ROC-AUC", nlp["roc_auc"])
    kpi[2].metric("Macro-F1", nlp["f1_macro"])

    st.image(str(FIG_DIR / "12_sentiment_terms.png"))
    share = nlp["late_vs_neg_share"]
    st.write(f"Negative share: **{share['late_neg_pct']}%** of late orders' comments "
             f"vs **{share['ontime_neg_pct']}%** for on-time orders.")

    text = st.text_area("Try a Portuguese review", "Produto chegou rápido e muito bom, recomendo!")
    if text:
        features = sentiment["vectorizer"].transform([text])
        st.metric("P(positive)", f"{float(sentiment['classifier'].predict_proba(features)[0, 1]) * 100:.0f}%")


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    orders = load_orders()
    metrics = json.load(open(OUT_DIR / "metrics.json"))
    late_bundle, sentiment = load_models()

    page = st.sidebar.radio("Page", PAGES)
    st.sidebar.caption("Olist Brazilian E-Commerce · 2016–2018")

    if page == "Overview":
        page_overview(orders, metrics)
    elif page == "Late-delivery model":
        page_late_model(orders, metrics, late_bundle)
    elif page == "Customer segments":
        page_segments(metrics)
    else:
        page_sentiment(metrics, sentiment)


main()
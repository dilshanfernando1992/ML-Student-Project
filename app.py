"""
CartSense — Real-Time Purchase-Intent Scoring
COM763 Advanced Machine Learning — Task 1 deployed artefact.

Run locally:  streamlit run app.py
Deployed on:  Streamlit Community Cloud
"""
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from features import build_features

st.set_page_config(page_title="CartSense — Purchase-Intent Scoring",
                   page_icon="🛒", layout="wide")

MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITORS = ["Returning_Visitor", "New_Visitor", "Other"]


@st.cache_resource
def load_bundle():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


try:
    BUNDLE = load_bundle()
except FileNotFoundError:
    st.error("model.pkl not found. Run Student_Project.ipynb to regenerate it, then commit it to the repo.")
    st.stop()

MODEL = BUNDLE["model"]
DEFAULT_T = float(BUNDLE["threshold"])
METRICS = BUNDLE["metrics"]

st.title("🛒 CartSense — Real-Time Purchase-Intent Scoring")
st.caption("Predicts whether a live browsing session will end in a purchase, so that retention "
           "budget is spent only on sessions the model believes are winnable. "
           "Trained on the UCI Online Shoppers Purchasing Intention dataset (12,330 sessions).")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Session signals")
    st.caption("These mirror the Google Analytics fields available in real time.")

    st.subheader("Page activity")
    admin = st.number_input("Administrative pages", 0, 30, 2)
    admin_d = st.number_input("Administrative dwell (s)", 0.0, 4000.0, 60.0, step=10.0)
    info = st.number_input("Informational pages", 0, 25, 0)
    info_d = st.number_input("Informational dwell (s)", 0.0, 3000.0, 0.0, step=10.0)
    prod = st.number_input("Product pages", 0, 750, 32)
    prod_d = st.number_input("Product dwell (s)", 0.0, 64000.0, 1200.0, step=50.0)

    st.subheader("Analytics metrics")
    bounce = st.slider("Bounce rate", 0.0, 0.2, 0.005, 0.001, format="%.3f")
    exit_r = st.slider("Exit rate", 0.0, 0.2, 0.020, 0.001, format="%.3f")
    pv = st.number_input("Page value (currency)", 0.0, 400.0, 12.0, step=1.0,
                         help="Google Analytics Page Value. See the leakage caveat on the "
                              "'Model & limitations' tab — this feature is the single "
                              "strongest driver of the prediction.")
    sday = st.select_slider("Special-day proximity", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 0.0)

    st.subheader("Context")
    month = st.selectbox("Month", MONTHS, index=8)
    vtype = st.selectbox("Visitor type", VISITORS, index=0)
    weekend = st.checkbox("Weekend session", value=False)
    os_ = st.selectbox("Operating system ID", list(range(1, 9)), index=1)
    browser = st.selectbox("Browser ID", list(range(1, 14)), index=1)
    region = st.selectbox("Region ID", list(range(1, 10)), index=2)
    traffic = st.selectbox("Traffic type ID", list(range(1, 21)), index=1)

raw = pd.DataFrame([dict(
    Administrative=admin, Administrative_Duration=admin_d, Informational=info,
    Informational_Duration=info_d, ProductRelated=prod, ProductRelated_Duration=prod_d,
    BounceRates=bounce, ExitRates=exit_r, PageValues=pv, SpecialDay=sday, Month=month,
    OperatingSystems=os_, Browser=browser, Region=region, TrafficType=traffic,
    VisitorType=vtype, Weekend=weekend)])

tab1, tab2, tab3 = st.tabs(["Score a session", "Batch scoring", "Model & limitations"])

# ---------------------------------------------------------------- tab 1
with tab1:
    thr = st.slider("Decision threshold", 0.05, 0.95, DEFAULT_T, 0.005,
                    help=f"Tuned to {DEFAULT_T:.3f} by maximising out-of-fold F1. Raise it for "
                         "higher precision (cheaper campaigns), lower it for higher recall "
                         "(catch more buyers).")
    prob = float(MODEL.predict_proba(build_features(raw))[0, 1])
    will_buy = prob >= thr

    c1, c2, c3 = st.columns([1.1, 1, 1.4])
    c1.metric("Purchase probability", f"{prob*100:.1f}%")
    c2.metric("Decision", "LIKELY BUYER" if will_buy else "UNLIKELY", f"threshold {thr:.3f}")
    lift = prob / 0.155
    c3.metric("Lift vs. base rate", f"{lift:.1f}×", "base rate = 15.5%")

    st.progress(min(prob, 1.0))

    if will_buy:
        st.success("**Recommended action — do not discount.** This session is already converting on "
                   "its own; an incentive here mostly gives margin away. Prioritise a frictionless "
                   "checkout and keep the session uninterrupted.")
    elif prob >= thr * 0.5:
        st.warning("**Recommended action — intervene.** Borderline session: this is where retention "
                   "spend earns its keep. Trigger live chat, free-delivery messaging, or an exit-intent offer.")
    else:
        st.info("**Recommended action — no spend.** Low intent. Serve organic content only; "
                "targeting this session is unlikely to repay the intervention cost.")

    with st.expander("Feature vector sent to the model"):
        st.dataframe(build_features(raw).T.rename(columns={0: "value"}), use_container_width=True)

# ---------------------------------------------------------------- tab 2
with tab2:
    st.markdown("Upload a CSV with the **18 raw UCI columns** (the `Revenue` column is optional and ignored) "
                "to score a whole day of traffic at once.")
    up = st.file_uploader("Session CSV", type="csv")
    if up is not None:
        try:
            d = pd.read_csv(up)
            probs = MODEL.predict_proba(build_features(d))[:, 1]
            out = d.copy()
            out["purchase_probability"] = probs.round(4)
            out["flagged_for_intervention"] = (probs >= DEFAULT_T).astype(int)
            a, b = st.columns(2)
            a.metric("Sessions scored", len(out))
            b.metric("Flagged as likely buyers", f"{int(out.flagged_for_intervention.sum())} "
                                                 f"({out.flagged_for_intervention.mean()*100:.1f}%)")
            st.dataframe(out.head(50), use_container_width=True)
            st.download_button("Download scored sessions",
                               out.to_csv(index=False).encode(),
                               "scored_sessions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not score that file: {e}")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.subheader("Final model — held-out test performance")
    st.dataframe(pd.DataFrame([{k: v for k, v in METRICS.items() if k != "Model"}]),
                 use_container_width=True, hide_index=True)
    st.caption(f"Model: tuned HistGradientBoostingClassifier · decision threshold {DEFAULT_T:.3f} · "
               f"scikit-learn {BUNDLE['sklearn_version']} · 2,441 unseen sessions.")

    st.subheader("Known limitations")
    st.markdown(
        "- **Page Value dependence / leakage risk.** Removing `PageValues` collapses PR-AUC from "
        "**0.753 to 0.374**. Google Analytics derives this metric partly from completed transactions, "
        "so early in a real session it is far weaker than it looks here. Treat the headline figures as "
        "an upper bound, and re-validate on live streamed data before trusting the model at cart entry.\n"
        "- **Single retailer, single year.** The data covers one e-commerce site over one year; "
        "conversion behaviour, seasonality and traffic mix will differ elsewhere.\n"
        "- **Not a customer-level tool.** Sessions are anonymous and independent. This model must not "
        "be used to profile identified individuals or to price differently per person.\n"
        "- **Threshold is a business decision.** F1-optimal (0.675) and value-optimal (~0.87) thresholds "
        "differ. The slider exists so that the operator, not the model, owns that trade-off.")

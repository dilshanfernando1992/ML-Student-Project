# CartSense — Purchase-Intent Prediction for E-Commerce Revenue Recovery

**COM763 Advanced Machine Learning — Task 1**

Predicts whether a live browsing session will end in a purchase, so retention budget
(discounts, live chat, exit-intent offers) is spent only on the sessions that are winnable.


## Results (held-out test, 2,441 unseen sessions)

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.50 | 0.8624 | 0.5378 | 0.8560 | 0.6606 | 0.9276 | 0.6907 |
| Random Forest | 0.50 | 0.9013 | 0.6890 | 0.6728 | 0.6808 | 0.9256 | 0.7316 |
| HistGradientBoosting | 0.50 | 0.8742 | 0.5706 | 0.7932 | 0.6637 | 0.9318 | 0.7421 |
| **HistGradientBoosting (tuned + threshold)** | **0.675** | **0.8988** | **0.6596** | **0.7304** | **0.6932** | **0.9351** | **0.7526** |

Primary metrics are **PR-AUC and F1**, not accuracy: the class is 15.5% positive, so a
do-nothing model scores 84.5% accuracy while flagging zero buyers.

## Known limitation 

Removing `PageValues` collapses PR-AUC from **0.753 → 0.374**. Google Analytics derives
Page Value partly from transaction revenue attributed to pages in the session, so it is
partially contaminated by the outcome it predicts. The headline figures are an
**upper bound**, not a deployment forecast: the model is strong late in a session and
much weaker at cart entry. See 4.2 of the report.

## Files

| File | Purpose |
|---|---|
| `Student_Project.ipynb` | Full pipeline: load → clean → EDA → train → evaluate → save |
| `features.py` | Feature engineering shared by training **and** serving (prevents train/serve skew) |
| `app.py` | Streamlit app: single-session scoring, batch CSV scoring, model card |
| `model.pkl` | Bundle: fitted pipeline + decision threshold + feature contract + version |
| `dataset.csv` | Snapshot of the UCI data used for this run |
| `requirements.txt` | Pinned runtime — `scikit-learn` is hard-pinned to match `model.pkl` |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

1. Push all files to a **public** repo named `ML-Student-Project`.
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub.
3. Select the repo, branch `main`, main file `app.py` → **Deploy**.
4. Wait ~2 minutes, copy the `*.streamlit.app` URL into the report.


## Reference

C. O. Sakar, S. O. Polat, M. Katircioglu, and Y. Kastro, "Real-time prediction of online
shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural
networks," *Neural Computing and Applications*, vol. 31, no. 10, pp. 6893–6908, 2019.

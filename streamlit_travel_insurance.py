import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix

# ---------- helper functions ----------
def get_final_estimator(pipe):
    """
    Ambil estimator akhir dari pipeline. Jika ada wrapper (CalibratedClassifier, etc),
    coba unwrap ke estimator yang punya coef_.
    """
    if hasattr(pipe, "named_steps"):
        final = pipe.steps[-1][1]
    else:
        final = pipe

    # unwrap common wrappers
    # calibrated classifier has attribute 'base_estimator_'
    if hasattr(final, "estimator_"):
        final = final.estimator_
    if hasattr(final, "base_estimator_"):
        final = final.base_estimator_
    return final

def get_transformer(pipe):
    """Coba ambil transformer/pra-proses dari pipeline (nama umum: 'transform', 'preprocessor', 'transformer')."""
    if not hasattr(pipe, "named_steps"):
        return None
    for name in ["transform", "transformer", "preprocessor", "preprocess"]:
        if name in pipe.named_steps:
            return pipe.named_steps[name]
    # fallback: kalau ada >1 step, ambil step pertama
    return pipe.steps[0][1] if len(pipe.steps) > 1 else None

def get_feature_names(transformer, input_df):
    """
    Dapatkan nama fitur hasil transformasi.
    Jika transformer punya get_feature_names_out -> pakai itu,
    else transform input_df dan kembalikan generic names sesuai shape.
    """
    # try get_feature_names_out (ColumnTransformer/OneHot etc)
    try:
        # some transformers require input to be passed to get_feature_names_out
        if hasattr(transformer, "get_feature_names_out"):
            try:
                names = transformer.get_feature_names_out()
            except TypeError:
                # get_feature_names_out(X) sometimes needs X
                names = transformer.get_feature_names_out(input_df)
            return list(names)
    except Exception:
        pass

    # fallback: transform and generate generic names
    try:
        X_tr = transformer.transform(input_df)
        n_cols = X_tr.shape[1]
        return [f"f_{i}" for i in range(n_cols)]
    except Exception:
        # ultimate fallback: original columns
        return list(input_df.columns)

def extract_coefficients(model_pipeline, input_df):
    """
    Return (feature_names, coefficients) or (None, None) if not available.
    """
    final_est = get_final_estimator(model_pipeline)
    transformer = get_transformer(model_pipeline)
    feature_names = get_feature_names(transformer, input_df) if transformer is not None else list(input_df.columns)

    # try coefficients
    if hasattr(final_est, "coef_"):
        coefs = final_est.coef_
        # multiclass or binary handling
        if coefs.ndim == 2:
            coefs = coefs[0]
        coefs = np.array(coefs)
        if len(coefs) == len(feature_names):
            return feature_names, coefs
        else:
            # mismatch length -> still return with truncated/expanded names as fallback
            min_len = min(len(feature_names), len(coefs))
            return feature_names[:min_len], coefs[:min_len]

    # if final estimator doesn't have coef_
    return None, None

# ---------- Streamlit app ----------
st.set_page_config(layout="wide")
st.title("✈️ Travel Insurance — Prediction & Interpretation")

# load pipeline (saved pipeline object)
MODEL_PATH = "travel_insurance_logreg.sav"
with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)

st.sidebar.header("Input features")

# Numerical Features
duration = st.sidebar.slider('Duration', min_value=0, max_value=547, value=50)
net_sales = st.sidebar.number_input('Net Sales', min_value=-357.5, max_value=666.0, value=42.0)
commision = st.sidebar.number_input('Commision', min_value=0.0, max_value=262.76, value=10.23)
age = st.sidebar.slider('Age', min_value=18, max_value=88, value=38)

# Categorical Features
agency = st.sidebar.selectbox('Agency', ['RAB', 'JZI', 'C2B', 'LWC', 'EPX', 'CWT', 'KML', 'TST', 'CCR', 'SSI', 'CBH', 'ART', 'TTW', 'ADM', 'CSR'])
agency_type = st.sidebar.radio('Agency Type', ['Airlines', 'Travel Agency'])
distribution_channel = st.sidebar.radio('Distribution Channel', ['Online', 'Offline'])

product_name = st.sidebar.selectbox('Product Name', [
    'Value Plan', 'Annual Gold Plan', 'Single Trip Travel Protect Gold', 'Cancellation Plan',
    'Bronze Plan', '1 way Comprehensive Plan', 'Single Trip Travel Protect Platinum', 'Basic Plan',
    '2 way Comprehensive Plan', 'Rental Vehicle Excess Insurance', 'Silver Plan', 'Annual Silver Plan',
    'Travel Cruise Protect', 'Single Trip Travel Protect Silver', 'Gold Plan', 'Comprehensive Plan',
    'Ticket Protector', 'Annual Travel Protect Gold', 'Child Comprehensive Plan', 'Premier Plan',
    'Annual Travel Protect Silver', 'Individual Comprehensive Plan', 'Spouse or Parents Comprehensive Plan',
    'Annual Travel Protect Platinum', '24 Protect', 'Travel Cruise Protect Family'
])

destination = st.sidebar.selectbox('Destination', [
    'BRUNEI DARUSSALAM', 'KOREA REPUBLIC OF', 'SINGAPORE', 'UNITED KINGDOM', 'CHINA', 'INDIA', 'THAILAND',
    'PHILIPPINES', 'SPAIN', 'HONG KONG', 'MALAYSIA', 'AUSTRALIA', 'IRELAND', 'UNITED STATES', 'MYANMAR',
    'FRANCE', 'INDONESIA', 'CAMBODIA', 'NEW ZEALAND', 'TAIWAN PROVINCE OF CHINA', 'JAPAN', 'GERMANY',
    'SOUTH AFRICA', 'NORWAY', 'ITALY', 'AUSTRIA', 'UNITED ARAB EMIRATES', 'SWITZERLAND', 'BRAZIL',
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC", 'NEPAL', 'CANADA', 'VIET NAM', 'BANGLADESH', 'SAUDI ARABIA',
    'JORDAN', 'MACAO', 'PORTUGAL', 'SWEDEN', 'DENMARK', 'PAKISTAN', 'MALTA', 'ISRAEL', 'GREECE', 'COLOMBIA',
    'MACEDONIA THE FORMER YUGOSLAV REPUBLIC OF', 'PERU', 'AZERBAIJAN', 'BELGIUM', 'OMAN', 'BAHRAIN', 'KENYA',
    'SRI LANKA', 'BENIN', 'NETHERLANDS', 'CROATIA', 'ICELAND', 'GUINEA', 'SLOVENIA', 'CZECH REPUBLIC',
    'PAPUA NEW GUINEA', 'PANAMA', 'TURKEY', 'BHUTAN', 'FRENCH POLYNESIA', 'HUNGARY', 'TUNISIA', 'FINLAND',
    'VANUATU', 'GHANA', 'MONGOLIA', 'MAURITIUS', 'MEXICO', 'QATAR', 'RUSSIAN FEDERATION', 'COSTA RICA',
    'VENEZUELA', 'MALDIVES', 'KUWAIT', 'UZBEKISTAN', 'MOROCCO', 'POLAND', 'GEORGIA', 'TANZANIA UNITED REPUBLIC OF',
    'BERMUDA', 'KYRGYZSTAN', 'ECUADOR', 'UKRAINE', 'ARGENTINA', 'EGYPT', 'ETHIOPIA', 'BELARUS', 'UGANDA', 'FIJI',
    'ROMANIA', 'ZAMBIA', 'GUADELOUPE', 'CYPRUS', 'KAZAKHSTAN', 'ANGOLA', 'BULGARIA', 'FAROE ISLANDS', 'TAJIKISTAN',
    'CHILE', 'LATVIA', 'GUAM', 'NORTHERN MARIANA ISLANDS', 'BOLIVIA', 'ZIMBABWE', 'CAMEROON', 'TURKMENISTAN',
    'LEBANON', 'ESTONIA', 'LITHUANIA', 'SERBIA', 'ARMENIA', 'LUXEMBOURG', 'TRINIDAD AND TOBAGO', 'NAMIBIA', 'GUYANA',
    'JAMAICA', 'REPUBLIC OF MONTENEGRO', 'SENEGAL', 'PUERTO RICO', 'CAYMAN ISLANDS', 'IRAN ISLAMIC REPUBLIC OF',
    'URUGUAY', 'NIGERIA', 'BOTSWANA', 'MALI', 'MOLDOVA REPUBLIC OF', 'SAMOA'
])

# threshold
threshold = st.sidebar.slider("Decision threshold (use tuned value)", min_value=0.01, max_value=0.99, value=0.55, step=0.01)

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([{
        "Agency": agency,
        "Agency Type": agency_type,
        "Distribution Channel": distribution_channel,
        "Product Name": product_name,
        "Destination": destination,
        "Duration": duration,
        "Net Sales": net_sales,
        "Commision": commision,
        "Age": age
    }])

    # Probability & prediction using custom threshold
    try:
        probs = pipeline.predict_proba(input_df)[0]
    except Exception as e:
        st.error(f"Error saat predict_proba: {e}")
        st.stop()

    prob_pos = probs[1]
    pred_label = int(prob_pos >= threshold)

    # Show colored UI: green if not claim (0), red if claim (1)
    if pred_label == 1:
        st.error(f"Prediksi: CLAIM  (Prob = {prob_pos:.3f})")
    else:
        st.success(f"Prediksi: TIDAK CLAIM  (Prob = {prob_pos:.3f})")

    # show probabilities
    st.write(pd.DataFrame({"Class": ["Not Claim (0)", "Claim (1)"], "Probability": probs}))

    # confusion & metrics example: (if you have holdout y_test and X_test)
    # st.write(...)  # optionally show metrics

    # ---------------- Interpretation: coefficients ----------------
    feat_names, coefs = extract_coefficients(pipeline, input_df)
    if feat_names is None or coefs is None:
        st.warning("Koefisien tidak tersedia untuk estimator terakhir (mungkin bukan linear model). "
                   "Pertimbangkan menggunakan SHAP atau Permutation Importance untuk interpretasi.")
    else:
        # build DataFrame for plotting
        df_imp = pd.DataFrame({"feature": feat_names, "coef": coefs})
        df_imp["abs"] = df_imp["coef"].abs()
        df_imp = df_imp.sort_values("abs", ascending=True)

        st.subheader("📉 Feature Importance")
        fig, ax = plt.subplots(figsize=(8, max(3, len(df_imp)*0.2)))
        ax.barh(df_imp["feature"], df_imp["coef"], color="tab:blue")
        ax.axvline(0, color="red", linestyle="--")
        ax.set_xlabel("Coefficient (log-odds)")
        ax.set_title("Logistic Regression coefficients (sorted by abs)")
        plt.tight_layout()
        st.pyplot(fig)

        # show top positive / negative
        top_pos = df_imp.sort_values("coef", ascending=False).head(10)
        top_neg = df_imp.sort_values("coef", ascending=True).head(10)
        st.subheader("Top positive contributors (increase probability → class 1)")
        st.dataframe(top_pos[["feature","coef"]].reset_index(drop=True))
        st.subheader("Top negative contributors (decrease probability → class 1)")
        st.dataframe(top_neg[["feature","coef"]].reset_index(drop=True))

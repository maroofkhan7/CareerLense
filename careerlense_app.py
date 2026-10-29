import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

st.set_page_config(
    page_title="CareerLens | Salary Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Global canvas and readable typography */
.stApp { background: #f3f6fb; color: #172033; }
[data-testid="stAppViewContainer"] { background: #f3f6fb; }
[data-testid="stHeader"] { background: rgba(243,246,251,.92); }
.block-container { max-width: 1380px; padding-top: 1.4rem; padding-bottom: 3rem; }
.main .block-container, [data-testid="stMain"] { color: #172033; }
.main p, .main label, .main li, .main span, .main small,
[data-testid="stMain"] p, [data-testid="stMain"] label { color: #344054; }
.main h1, .main h2, .main h3, .main h4,
[data-testid="stMain"] h1, [data-testid="stMain"] h2,
[data-testid="stMain"] h3, [data-testid="stMain"] h4 { color: #152238 !important; }
.hero {
    padding: 30px 34px; border-radius: 22px; color: white;
    background: linear-gradient(120deg, #102044 0%, #24529a 54%, #0d9488 100%);
    margin-bottom: 24px; box-shadow: 0 12px 30px rgba(26,55,105,.16);
}
.hero h1 { color: white !important; font-size: 2.35rem; margin-bottom: .25rem; letter-spacing: -.5px; }
.hero p { color: #e5efff !important; font-size: 1.05rem; margin-bottom: 0; }
[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e3eaf4; padding: 16px 18px; border-radius: 15px; box-shadow: 0 4px 15px rgba(20,40,80,.045); }
[data-testid="stMetricLabel"] p { color: #52627a !important; font-weight: 600; }
[data-testid="stMetricValue"] { color: #172b4d !important; }
.section-card { background: white; border: 1px solid #e1e8f2; border-radius: 18px; padding: 22px; margin-bottom: 16px; box-shadow: 0 6px 22px rgba(20,40,80,.05); }
[data-testid="stTabs"] button { color: #344054 !important; font-weight: 650; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #176b68 !important; }
[data-testid="stAlert"] { border-radius: 12px; }
div.stButton > button { border-radius: 11px; font-weight: 700; min-height: 3rem; border: 0; box-shadow: 0 4px 12px rgba(20,70,130,.13); }
[data-testid="stDataFrame"] { border: 1px solid #e3eaf4; border-radius: 12px; overflow: hidden; }
hr { border-color: #dce4ef; }
</style>
<div class="hero">
  <h1>💼 CareerLens</h1>
  <p>Job Salary Intelligence • Regression and Classification in one interactive workspace</p>
</div>
""", unsafe_allow_html=True)

DATA_PATH = "job_salary_prediction_dataset.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {
        "job_title": ["job_title", "job_role", "title"],
        "experience_years": ["experience_years", "years_experience", "experience"],
        "education_level": ["education_level", "education", "education_required"],
        "skills_count": ["skills_count", "number_of_skills"],
        "industry": ["industry", "industry_type"],
        "company_size": ["company_size", "organization_size"],
        "location": ["location", "company_location", "country"],
        "certifications": ["certifications", "certification_count"],
        "salary": ["salary", "annual_salary", "salary_usd", "salary_in_usd"]
    }
    rename = {}
    for standard, options in aliases.items():
        for option in options:
            if option in df.columns:
                rename[option] = standard
                break
    df = df.rename(columns=rename)
    required = [
        "job_title", "experience_years", "education_level", "skills_count",
        "industry", "company_size", "location",
        "certifications", "salary"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Available columns: {df.columns.tolist()}"
        )
    df = df[required].copy()
    for col in ["experience_years", "skills_count", "certifications", "salary"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    cat_cols = ["job_title", "education_level", "industry",
                "company_size", "location"]
    for col in cat_cols:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    df = df.dropna(subset=["salary", "job_title"])
    df = df[df["salary"] > 0]
    df = df.drop_duplicates()
    return df

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Dataset file '{DATA_PATH}' not found. Download the CSV from Kaggle "
        "and place it in the same folder as this app.py file."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

cat_features = [
    "job_title", "education_level", "industry",
    "company_size", "location"
]
num_features = ["experience_years", "skills_count", "certifications"]
features = cat_features + num_features
target = "salary"

if len(df) < 30:
    st.warning("This dataset has fewer than 30 usable rows; model metrics may be unstable.")

# Model preparation and held-out evaluation
preprocessor = ColumnTransformer([
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_features),
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_features)
])

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

@st.cache_resource
def train_models(_X_train, _y_train, _X_test, _y_test, _df):
    # Separate pipelines avoid sharing fitted preprocessing objects.
    reg_prep = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_features),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_features)
    ])
    reg = Pipeline([
        ("preprocessor", reg_prep),
        ("model", LinearRegression())
    ])
    reg.fit(_X_train, _y_train)
    reg_pred = reg.predict(_X_test)

    q1, q2 = _df[target].quantile([1/3, 2/3]).tolist()
    y_train_class = pd.cut(
        _y_train, bins=[-np.inf, q1, q2, np.inf],
        labels=["Low", "Medium", "High"], include_lowest=True
    )
    y_test_class = pd.cut(
        _y_test, bins=[-np.inf, q1, q2, np.inf],
        labels=["Low", "Medium", "High"], include_lowest=True
    )
    cls_prep = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_features),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_features)
    ])
    clf = Pipeline([
        ("preprocessor", cls_prep),
        ("model", LogisticRegression(
            max_iter=1500, class_weight="balanced", solver="lbfgs"
        ))
    ])
    clf.fit(_X_train, y_train_class)
    cls_pred = clf.predict(_X_test)
    return reg, clf, reg_pred, y_test, cls_pred, y_test_class, q1, q2

reg_model, cls_model, reg_pred, y_test_reg, cls_pred, y_test_cls, q1, q2 = train_models(
    X_train, y_train, X_test, y_test, df
)

# Sidebar: inputs use meaningful job-related fields only.
with st.sidebar:
    st.markdown("## 🎛️ Candidate & Role Profile")
    st.caption("Build a job profile to explore salary estimates.")
    job_title = st.selectbox("Job Title", sorted(df["job_title"].dropna().unique()))
    experience = st.slider(
        "Years of Experience",
        min_value=int(max(0, np.floor(df["experience_years"].min()))),
        max_value=int(np.ceil(df["experience_years"].max())),
        value=int(np.clip(np.median(df["experience_years"]), 0,
                          max(0, np.ceil(df["experience_years"].max()))))
    )
    education = st.selectbox(
        "Education Level", sorted(df["education_level"].dropna().unique())
    )
    industry = st.selectbox("Industry", sorted(df["industry"].dropna().unique()))
    company_size = st.selectbox(
        "Company Size", sorted(df["company_size"].dropna().unique())
    )
    location = st.selectbox("Location", sorted(df["location"].dropna().unique()))
    skills_count = st.slider(
        "Number of Skills", int(max(0, df["skills_count"].min())),
        int(df["skills_count"].max()),
        int(np.clip(df["skills_count"].median(),
                    max(0, df["skills_count"].min()), df["skills_count"].max()))
    )
    certs = st.slider(
        "Certifications", int(max(0, df["certifications"].min())),
        int(df["certifications"].max()),
        int(np.clip(df["certifications"].median(),
                    max(0, df["certifications"].min()), df["certifications"].max()))
    )
    predict_clicked = st.button("✨ Predict Salary & Class", type="primary", use_container_width=True)

st.markdown("### 📊 Dataset at a glance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Usable job records", f"{len(df):,}")
m2.metric("Distinct job titles", f"{df['job_title'].nunique():,}")
m3.metric("Median salary", f"{df['salary'].median():,.0f}")
m4.metric("Input features", f"{len(features)}")

st.markdown("---")
st.markdown("### Explore the prediction workspace")
st.caption("Choose a model tab below to estimate salary, classify a salary band, or explore market trends.")
tab1, tab2, tab3 = st.tabs(["💰 Salary Estimate", "🧭 Salary Class", "📈 Market Insights"])

input_row = pd.DataFrame([{
    "job_title": job_title,
    "experience_years": experience,
    "education_level": education,
    "skills_count": skills_count,
    "industry": industry,
    "company_size": company_size,
    "location": location,
    "certifications": certs
}])

with tab1:
    st.markdown("#### Linear Regression · Numerical salary prediction")
    st.write("Estimate the dataset's salary target using the selected job profile.")
    if predict_clicked:
        estimate = float(reg_model.predict(input_row)[0])
        estimate = max(0, estimate)
        a, b = st.columns([1.2, 1])
        with a:
            st.markdown("**Estimated salary (dataset units)**")
            st.markdown(
                f"<div style='background:#ffffff; border:1px solid #dce5f1; "
                f"border-radius:16px; padding:24px 22px; margin:8px 0 12px 0;'>"
                f"<div style='font-size:2.35rem; font-weight:750; color:#0f766e; "
                f"line-height:1.25;'>{estimate:,.0f}</div></div>",
                unsafe_allow_html=True
            )
            st.caption("Interpret currency and pay period according to the dataset documentation.")
        with b:
            st.metric("Difference from dataset median",
                      f"{estimate - df['salary'].median():+,.0f}")
    else:
        st.info("Complete the profile in the sidebar and click **Predict Salary & Class**.")
    with st.expander("How to interpret this result"):
        st.write("Linear Regression predicts a continuous numerical value. It learns average relationships from the training data; it is an estimate, not a guaranteed offer or individual compensation assessment.")

with tab2:
    st.markdown("#### Logistic Regression · Salary category prediction")
    st.write(f"Categories are based on salary terciles in the dataset (cutoffs: {q1:,.0f} and {q2:,.0f}).")
    if predict_clicked:
        predicted_class = cls_model.predict(input_row)[0]
        probs = cls_model.predict_proba(input_row)[0]
        class_names = cls_model.named_steps["model"].classes_
        a, b = st.columns([1, 1.2])
        with a:
            st.metric("Predicted Salary Class", str(predicted_class))
            st.metric("Test accuracy", f"{accuracy_score(y_test_cls, cls_pred):.1%}")
        with b:
            prob_df = pd.DataFrame({
                "Salary Class": class_names,
                "Estimated Probability": probs
            }).set_index("Salary Class")
            st.bar_chart(prob_df)
    else:
        st.info("Click **Predict Salary & Class** to see the predicted salary category and probability estimates.")

with tab3:
    st.markdown("#### Explore the dataset")
    left, right = st.columns(2)
    with left:
        job_stats = df.groupby("job_title", as_index=False)["salary"].median().sort_values("salary", ascending=False).head(12)
        fig = px.bar(job_stats, x="salary", y="job_title", orientation="h",
                     title="Highest Median Salary by Job Title",
                     labels={"salary": "Median salary", "job_title": "Job title"},
                     color="salary", color_continuous_scale="Teal")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=480)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        exp_stats = df.groupby("experience_years", as_index=False)["salary"].median()
        fig2 = px.line(exp_stats, x="experience_years", y="salary",
                       markers=True, title="Median Salary vs Experience",
                       labels={"experience_years": "Years of experience", "salary": "Median salary"})
        fig2.update_layout(height=480)
        st.plotly_chart(fig2, use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        industry_stats = df.groupby("industry", as_index=False)["salary"].median().sort_values("salary", ascending=False)
        fig3 = px.bar(industry_stats, x="industry", y="salary",
                      title="Median Salary by Industry", labels={"industry": "Industry", "salary": "Median salary"})
        fig3.update_layout(xaxis_tickangle=-35, height=420)
        st.plotly_chart(fig3, use_container_width=True)
    with right2:
        st.markdown("**Model evaluation (held-out test data)**")
        st.dataframe(pd.DataFrame({
            "Metric": ["Regression R²", "Classification Accuracy"],
            "Value": [
                f"{r2_score(y_test_reg, reg_pred):.3f}",
                f"{accuracy_score(y_test_cls, cls_pred):.1%}"
            ]
        }), hide_index=True, use_container_width=True)

with st.expander("ℹ️ Model notes & responsible use"):
    st.write("""
    - **Linear Regression** predicts numerical salary.
    - **Logistic Regression** predicts Low / Medium / High salary class.
    - Inputs are job-related: role, experience, education, skills, industry, company size, location, work arrangement, and certifications.
    - Sex, race, and ethnic origin are not used.
    - Results depend on the dataset's coverage, salary units, and data quality. Do not use this educational demo as the sole basis for compensation decisions.

    """)

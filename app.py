
import streamlit as st
import joblib
import re
import html
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="SentiView | AI Sentiment Analyzer",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM DESIGN
# ============================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #0B1020;
    color: #F8FAFC;
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #111A30;
    border-right: 1px solid #25304A;
}

[data-testid="stSidebar"] p {
    color: #AAB6D0;
}

.hero {
    background: linear-gradient(120deg, #182448, #25205A, #172A4B);
    border: 1px solid #384477;
    border-radius: 24px;
    padding: 42px 35px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 15px 50px rgba(0,0,0,.22);
}

.hero-tag {
    display: inline-block;
    background: #30366A;
    color: #C5CBFF;
    padding: 8px 17px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.hero h1 {
    font-size: 43px;
    font-weight: 800;
    margin: 19px 0 10px;
    color: white;
}

.hero h1 span {
    color: #A5A7FF;
}

.hero p {
    color: #C0CBE2;
    font-size: 16px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 8px;
}

.section-subtitle {
    color: #8998B5;
    font-size: 14px;
    margin-bottom: 20px;
}

.review-card {
    background: #141F35;
    border: 1px solid #2A3858;
    padding: 25px;
    border-radius: 20px;
    margin-bottom: 20px;
}

.stTextArea textarea {
    background: #0E172A !important;
    color: white !important;
    border: 1px solid #364568 !important;
    border-radius: 13px !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
}

.stTextArea textarea:focus {
    border: 1px solid #8885FF !important;
    box-shadow: 0 0 0 1px #8885FF !important;
}

.stButton > button {
    background: linear-gradient(100deg, #655CF6, #9B6BFF);
    color: white;
    border: none;
    border-radius: 12px;
    height: 50px;
    font-size: 16px;
    font-weight: 700;
    transition: .3s;
}

.stButton > button:hover {
    background: linear-gradient(100deg, #8077FF, #B18AFF);
    color: white;
    border: none;
    transform: translateY(-2px);
}

.result {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-top: 22px;
}

.positive {
    background: linear-gradient(120deg, #102E2B, #153B37);
    border: 1px solid #287D69;
}

.negative {
    background: linear-gradient(120deg, #351C2A, #421F32);
    border: 1px solid #A34C68;
}

.result-label {
    font-size: 12px;
    letter-spacing: 2px;
    color: #AAB6D0;
    font-weight: 700;
}

.result h1 {
    font-size: 36px;
    margin: 12px 0;
}

.positive h1 {
    color: #57E6B1;
}

.negative h1 {
    color: #FF829A;
}

.result p {
    color: #C3CEE0;
    font-size: 14px;
}

.info-card {
    background: #141F35;
    border: 1px solid #293957;
    border-radius: 16px;
    padding: 19px;
    text-align: center;
}

.info-card h2 {
    color: #B2A6FF;
    font-size: 25px;
    margin: 5px 0;
}

.info-card p {
    color: #91A0BB;
    font-size: 12px;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #7168FF, #B17BFF);
}

.footer {
    text-align: center;
    color: #75839F;
    font-size: 12px;
    padding-top: 25px;
    border-top: 1px solid #29334C;
    margin-top: 45px;
}

</style>
""", unsafe_allow_html=True)


# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Unable to load model: {e}")


# ============================================
# TEXT CLEANING
# ============================================

stop_words = set(ENGLISH_STOP_WORDS) - {
    "not", "no", "nor", "never"
}

def clean_text(text):

    text = html.unescape(str(text))
    text = text.lower()

    contractions = {
        "isn't": "is not",
        "wasn't": "was not",
        "aren't": "are not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "can't": "can not",
        "couldn't": "could not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "won't": "will not"
    }

    for old, new in contractions.items():
        text = text.replace(old, new)

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ============================================
# SIDEBAR
# ============================================

with st.sidebar:

    st.markdown("## ✦ SentiView")

    st.markdown(
        "<p style='color:#8C9AB7;'>AI-Powered Movie Review Analysis</p>",
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### ✨ About the Project")

    st.write(
        "SentiView uses Machine Learning "
        "to understand movie reviews and "
        "identify their emotional polarity."
    )

    st.divider()

    st.markdown("### 🧠 Technology")

    st.markdown("""
    - Python
    - TF-IDF
    - Logistic Regression
    - Scikit-learn
    - Streamlit
    """)

    st.divider()

    st.markdown("### 📊 Model Information")

    st.markdown("""
    **Model:** Logistic Regression

    **Dataset:** IMDb Movie Reviews

    **Task:** Binary Classification

    **Classes:** Positive / Negative
    """)

    st.divider()

    st.caption("SentiView | Machine Learning Project")



# ============================================
# HERO SECTION
# ============================================

st.html("""
<div class="hero">
    <div class="hero-tag">
        ✦ ARTIFICIAL INTELLIGENCE
    </div>

    <h1>
        Discover the <span>Emotion</span><br>
        Behind Every Review
    </h1>

    <p>
        Advanced sentiment analysis powered by
        Logistic Regression and Natural Language Processing.
    </p>

    <div style="
        margin-top:25px;
        color:#C5CBFF;
        font-size:13px;
        letter-spacing:1px;
    ">
        ✦ SMART ANALYSIS &nbsp; | &nbsp;
        ✦ MACHINE LEARNING &nbsp; | &nbsp;
        ✦ NLP
    </div>
</div>
""")


# ============================================
# STATISTICS
# ============================================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <h2>90.06%</h2>
        <p>MODEL ACCURACY</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <h2>90.20%</h2>
        <p>F1 SCORE</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <h2>49,582</h2>
        <p>MOVIE REVIEWS</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# ============================================
# REVIEW INPUT
# ============================================

st.markdown(
    '<div class="section-title">✍️ Analyze Your Movie Review</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Enter a movie review below and let AI discover its sentiment.'
    '</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    st.markdown("**Your Review**")

    examples = {
        "Choose an example...": "",
        "Positive example": (
            "This movie was absolutely amazing! "
            "The acting was wonderful and the story was beautiful. "
            "I really loved every moment."
        ),
        "Negative example": (
            "This movie was boring and terrible. "
            "The acting was awful and the story was disappointing. "
            "I hated it."
        )
    }

    selected = st.selectbox(
        "Try an example",
        list(examples.keys())
    )

    review = st.text_area(
        "Movie Review",
        value=examples[selected],
        height=170,
        placeholder="Write or paste your movie review here..."
    )

    st.caption("Write your review in English for best results.")

    analyze = st.button(
        "✦  ANALYZE SENTIMENT",
        use_container_width=True
    )
# # =====================================
# # MODEL PERFORMANCE COMPARISON
# # =====================================

# st.markdown("## 📊 Model Performance Comparison")

# try:
#     results = pd.read_csv("model_comparison.csv")

#     st.dataframe(results, use_container_width=True)

#     model_col = results.columns[0]
#     results = results.set_index(model_col)

#     metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

#     available_metrics = [
#         col for col in results.columns
#         if col.strip().lower() in [m.lower() for m in metrics]
#     ]

#     if available_metrics:
#         st.bar_chart(
#             results[available_metrics],
#             use_container_width=True
#         )
#     else:
#         st.warning("Performance metrics columns not found.")
# except FileNotFoundError:
#     st.warning("model_comparison.csv not found.")

# ============================================
# PREDICTION
# ============================================

if analyze:

    if not model_loaded:

        st.error("Please check your model file.")

    elif not review.strip():

        st.warning("Please enter a movie review first.")

    else:

        cleaned_review = clean_text(review)

        if not cleaned_review:

            st.warning("Please enter a valid review.")

        else:

            try:

                prediction = model.predict(
                    [cleaned_review]
                )[0]

                probabilities = model.predict_proba(
                    [cleaned_review]
                )[0]

                confidence = max(probabilities) * 100

                label = str(prediction).strip().lower()

                positive_labels = {"1", "positive", "pos"}
                negative_labels = {"0", "negative", "neg"}

                if label in positive_labels:
                    sentiment = "Positive"
                    icon = "😊"
                    css_class = "positive"
                    description = "This review expresses a positive opinion."
                elif label in negative_labels:
                    sentiment = "Negative"
                    icon = "☹️"
                    css_class = "negative"
                    description = "This review expresses a negative opinion."
                else:
                    st.error(
                        f"Unknown model output label: {prediction}"
                    )
                    st.stop()

                st.markdown("---")

                st.markdown(
                    '<div class="section-title">'
                    '📈 Sentiment Analysis Result'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="result {css_class}">
                        <div class="result-label">
                            PREDICTED SENTIMENT
                        </div>
                        <h1>{icon} {sentiment.upper()}</h1>
                        <p>{description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("### Confidence Score")

                st.progress(
                    int(round(confidence))
                )

                m1, m2 = st.columns(2)

                with m1:
                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2f}%"
                    )

                with m2:
                    st.metric(
                        "Predicted Class",
                        sentiment
                    )

                # Display class probabilities

                st.markdown("### Class Probabilities")

                classes = list(model.classes_)

                for class_name, probability in zip(
                    classes, probabilities
                ):

                    class_label = str(class_name).title()

                    st.write(
                        f"**{class_label}** — {probability*100:.2f}%"
                    )

                    st.progress(
                        int(round(probability * 100))
                    )

            except Exception as e:

                st.error(f"Prediction error: {e}")


# ============================================
# FOOTER
# ============================================

st.markdown("""
<div class="footer">
    <p>✦ SENTIVIEW — AI SENTIMENT ANALYZER</p>
    <p>Powered by Machine Learning | Logistic Regression</p>
    <p>IMDb Movie Reviews Dataset</p>
</div>
""", unsafe_allow_html=True)


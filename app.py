import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# ----------------------------
# Load model + tokenizer
# ----------------------------
@st.cache_resource
def load_model():
    model_path = "bert_imdb_state_dict.pt"   # your weights file

    # 1. Load tokenizer normally from pretrained
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # 2. Create model architecture manually
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    # 3. Load state dict
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# ---------------------------------
# Prediction Function
# ---------------------------------
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        label_id = torch.argmax(probs).item()
        confidence = probs[0][label_id].item()

    label = "Positive" if label_id == 1 else "Negative"
    return label, confidence


# ---------------------------------
# Modern UI Styling
# ---------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Classifier",
    page_icon="🎬",
    layout="centered",
)

st.markdown("""
    <style>
        body {
            background-color: #0e1117;
        }
        .title {
            font-size: 3rem;
            color: #00c3ff;
            text-align: center;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #bbbbbb;
            font-size: 1.1rem;
            margin-bottom: 40px;
        }
        .result-box {
            padding: 20px;
            border-radius: 12px;
            margin-top: 25px;
            font-size: 1.5rem;
            text-align: center;
            font-weight: bold;
        }
        .positive {
            background-color: #1b4332;
            color: #95d5b2;
            border: 2px solid #2d6a4f;
        }
        .negative {
            background-color: #5c1a1a;
            color: #f5b7b1;
            border: 2px solid #922b21;
        }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------
# Web Layout
# ---------------------------------
st.markdown("<div class='title'>IMDB Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter a movie review and let BERT predict the sentiment!</div>", unsafe_allow_html=True)

review = st.text_area("Enter your movie review here:", height=180)

if st.button("Predict Sentiment"):
    if len(review.strip()) == 0:
        st.warning("Please enter text before predicting.")
    else:
        label, conf = predict_sentiment(review)

        css_class = "positive" if label == "Positive" else "negative"

        st.markdown(
            f"<div class='result-box {css_class}'>Prediction: {label}<br>Confidence: {conf:.4f}</div>",
            unsafe_allow_html=True
        )

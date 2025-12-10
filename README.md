```markdown
# 🎬 IMDB Movie Review Sentiment Analysis using BERT + Streamlit

A complete end-to-end **Sentiment Analysis** project using **BERT (Bidirectional Encoder Representations from Transformers)** fine-tuned on the **IMDB 50K Movie Reviews Dataset**.

This repository includes:

- Training scripts (PyTorch + HuggingFace Transformers)
- Fine-tuned BERT model weights (`bert_imdb_state_dict.pt`)
- A modern interactive **Streamlit web app** for real-time sentiment prediction
- Requirements & setup instructions

---

## ⚡ Project Features

- Trained BERT model for **binary sentiment classification**
- Predicts **Positive** or **Negative** sentiment
- Clean and modern UI built with **Streamlit**
- Accepts custom user input and shows:
  - Prediction label
  - Model confidence score
- Easy deployment (Streamlit Cloud / HuggingFace Spaces / Render)

---

## 📂 Project Structure

```

sentiment_app/
│── app.py                      # Streamlit UI
│── bert_imdb_state_dict.pt     # Fine-tuned BERT weights
│── requirements.txt            # Dependencies
│── README.md                   # Documentation

````

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit web app

```bash
streamlit run app.py
```

The app will launch at:

```
http://localhost:8501
```

---

## 🧠 Model Details

* **Base model:** `bert-base-uncased`
* **Fine-tuned on:** Kaggle IMDB 50K Movie Reviews Dataset
* **Task:** Binary sentiment classification (Positive / Negative)
* **Frameworks used:**

  * PyTorch
  * HuggingFace Transformers
  * Streamlit for UI

The model was trained in Google Colab and saved as:

```
bert_imdb_state_dict.pt
```

This file stores the trained weights and is loaded in `app.py`.

---

## 🎨 Streamlit Web App Preview

> Replace these placeholders with real screenshots after running your app.

![App Screenshot 1](screenshots/app_home.png)
![App Screenshot 2](screenshots/app_prediction.png)

---

## 📝 Sample Inputs for Testing

Try these in the app:

**Positive examples**

* “This movie completely exceeded my expectations. The story was emotional and engaging.”
* “Great performances and a beautiful soundtrack. I would watch it again.”

**Negative examples**

* “The plot was boring and predictable. I couldn't wait for it to end.”
* “Terrible pacing, weak writing, and no memorable moments.”

---

## 📦 Deployment Guide

You can deploy the app easily on:

### ▶ Streamlit Cloud (Recommended)

1. Push your repo to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Select your repo
4. Deploy
5. App runs instantly online

### ▶ HuggingFace Spaces

* Add:

  * `app.py`
  * `requirements.txt`
  * `bert_imdb_state_dict.pt`
* Set SDK to **Streamlit**
* Deploy

### ▶ Render / Railway / Docker

* Let me know if you want a Dockerfile — I can generate one.



## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue to discuss what you’d like to modify.



## ⭐ Show Support

If you found this project helpful, please consider giving the repository a **star** on GitHub!



## 📬 Contact

For questions or feature requests, feel free to reach out or open an issue.

```

```

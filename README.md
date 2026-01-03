

# Sentiment Analysis Web Application (Flask)

This project is a **Sentiment Analysis web application built using Flask**, aimed at understanding sentiment from different input sources such as **text and URLs**.
The project is designed with scalability in mind, where support for **audio, video, and image-based sentiment analysis** can be added in the future.

> **Current focus:** Text and URL-based sentiment analysis
> **Future scope:** Media sentiment analysis (audio, video, images)

---

## 🔍 Project Overview

The application allows users to:

* Enter **text** or a **URL**
* Analyze sentiment using **pre-trained NLP models**
* View results as:

  * Prominent sentiment
  * Positive / Neutral / Negative scores
  * Visual representation (charts)

The goal of this project is to **understand end-to-end NLP workflows**, model integration, and web deployment using Flask.

---

## 🛠️ Technologies Used

* **Backend:** Python, Flask
* **NLP:** Hugging Face Transformers (RoBERTa-based model)
* **ML Frameworks:** PyTorch / TensorFlow (as required)
* **Frontend:** HTML, CSS, JavaScript
* **Visualization:** Charts for sentiment distribution

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/edwinrlambert/Sentiment-Analysis-Using-Flask.git
cd Sentiment-Analysis-Using-Flask
```

---

### 2️⃣ Create and Activate a Virtual Environment

**Install virtualenv (if not installed):**

```bash
pip install virtualenv
```

**Create virtual environment:**

```bash
virtualenv sentiment-analysis-env
```

**Activate environment (Windows):**

```bash
sentiment-analysis-env/Scripts/activate
```

---

### 3️⃣ Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Install Hugging Face Transformers (if required)

Depending on the backend you prefer:

```bash
pip install transformers[torch]
```

or

```bash
pip install transformers[tf-cpu]
```

---

### 5️⃣ Environment Variables Setup

Create a `.env` file to specify cache directories (important for large transformer models):

```env
HUGGINGFACE_CACHE_DIR=your_huggingface_cache_path
TORCH_CACHE_DIR=your_torch_cache_path
```

(Example provided in `.env.example`)

---

### 6️⃣ Run the Application

```bash
python app.py
```

The Flask server will start at:

```
http://127.0.0.1:5000
```

---

## 🧪 Application Features

### ✅ Text Sentiment Analysis

* Users input text
* Model predicts sentiment
* Displays:

  * Prominent sentiment
  * Confidence scores
  * Visualization chart

---

### ✅ URL Sentiment Analysis

* Extracts **text content from a webpage**
* Performs sentiment analysis on the extracted text
* Useful for:

  * Article analysis
  * Blog sentiment understanding

---

### ⚠️ Model Limitation Handling

Since the RoBERTa model supports a **maximum of ~510 tokens**, longer inputs are:

* Split into chunks
* Analyzed separately
* Final sentiment calculated as an **average**

> This may sometimes cause slight context loss, which is a known limitation.

---

## 🚧 Future Enhancements

* **Audio & Video Sentiment Analysis**

  * Transcribe speech → analyze text sentiment
* **Image-Based Sentiment**

  * Facial emotion detection
  * Object-based emotional cues
* Combine **audio + visual + textual sentiment** for videos
* Explore libraries like **DeepFace** for emotion recognition

This project is actively used as a **learning platform** to explore advanced NLP and multimodal sentiment analysis.

---

## 📌 Key Learning Outcomes

* End-to-end NLP pipeline implementation
* Practical use of Hugging Face transformer models
* Flask-based deployment of ML models
* Handling real-world limitations like input size and context loss

---

## 📄 License

This project is open-source and available under the **MIT License**.


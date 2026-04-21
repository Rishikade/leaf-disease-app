# 🌿 LeafSense AI — Plant Disease Detection System

An end-to-end deep learning project that detects plant leaf diseases using image classification and provides intelligent assistance through an interactive web application.

---

## 🚀 Live Features

* 📸 Upload or capture leaf images using camera
* 🧠 AI-based disease prediction (Deep Learning model)
* 📊 Confidence score with top-3 predictions
* 🌿 Disease explanation and treatment suggestions
* 🤖 Built-in chatbot for plant-related queries
* 📱 Responsive and clean web UI

---

## 🧠 Tech Stack

* **Python**
* **TensorFlow / Keras**
* **OpenCV**
* **Streamlit**
* **NumPy & PIL**

---

## 📂 Project Structure

```
leaf-disease-detection-ai/
│
├── app.py
├── train.py
├── requirements.txt
├── model/
│   └── leaf_model.h5
├── accuracy.png
├── confusion_matrix.png
└── README.md
```

---

## ⚙️ How It Works

1. User uploads or captures a leaf image
2. Image is preprocessed (resize + normalization)
3. CNN model predicts disease class
4. App displays:

   * Prediction result
   * Confidence score
   * Top 3 predictions
5. Chatbot assists with disease queries and prevention

---

## 📈 Model Details

* Input Size: **224x224**
* Model Type: **CNN (Deep Learning)**
* Classes:

  * Early Blight
  * Late Blight
  * Healthy

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix visualization

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📌 Key Highlights

* End-to-end ML pipeline (training → evaluation → deployment)
* Real-world problem solving (agriculture AI)
* Interactive UI with camera support
* Data visualization (confidence + prediction bars)

---

## 🔮 Future Improvements

* Mobile app deployment 📱
* Voice-enabled chatbot 🎤
* Multi-language support 🌍
* Real-time detection using video stream

---

## 👩‍💻 Author

**Rishika De**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to contribute!

🌿 AI Leaf Disease Detection Web App

This project is a deep learning-based web application that detects plant leaf diseases using image classification. It also includes a smart AI chatbot for plant-related queries.

---

🚀 Features

- 📸 Upload or capture leaf images
- 🧠 AI-based disease prediction (CNN model)
- 📊 Displays confidence score
- 🔝 Top 3 predictions
- 🌿 Disease information and solutions
- 🤖 AI chatbot for plant queries
- 📱 Mobile-friendly web app

---

🧠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- NumPy
- PIL
- OpenAI API (for chatbot)

---

📂 Project Structure

leaf-disease-project/
│
├── app.py
├── train.py
├── requirements.txt
├── model/
│   └── leaf_model.h5
└── dataset/ (not included in repo)

---

⚙️ Installation

1. Clone the repository:

git clone https://github.com/Rishikade/leaf-disease-app.git

2. Navigate to project folder:

cd leaf-disease-app

3. Install dependencies:

pip install -r requirements.txt

---

▶️ Run the App

streamlit run app.py

---

🔑 API Setup (Important)

- Get your API key from OpenAI
- Replace in app.py:

client = OpenAI(api_key="YOUR_API_KEY_HERE")

---

📸 How It Works

1. Upload or capture a leaf image
2. Model processes the image
3. Predicts disease category
4. Shows confidence and suggestions
5. Chatbot answers plant-related queries

---

📈 Future Improvements

- Real-time camera detection
- Mobile app (Android/iOS)
- Voice-based chatbot
- Multi-language support

---

💯 Project Highlights

- End-to-end AI project
- Real-world application
- Web + ML + Chatbot integration

---

📌 Note

- Dataset is not included due to size
- Model file must be present in "/model" folder

---

👩‍💻 Author

Developed by Rishika De

---


# 🌱 Crop & Fertilizer Recommendation System

This project is a **Machine Learning powered web application** that recommends the most suitable crop and fertilizer based on soil nutrients and weather conditions.  
It uses two ML models:
- **Crop Recommendation Model** – predicts the best crop based on soil nutrients and environmental data.  
- **Fertilizer Recommendation Model** – suggests the optimal fertilizer for the predicted crop.

---

## 🚀 Features
- Input soil nutrients (N, P, K), pH, temperature, humidity, rainfall, soil moisture, and soil type.  
- Get the **best crop recommendation**.  
- Get the **most suitable fertilizer recommendation**.  
- Simple and interactive **web interface** built with Streamlit.  

---

## 🛠️ Installation & Usage
1. Clone this repository:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<your-space-name>
   cd <your-space-name>


2. Install dependencies:

pip install -r requirements.txt


3. Run the app locally:

streamlit run app.py


4. Deploy easily on Hugging Face Spaces with Streamlit SDK.

📂 Project Structure
├── app.py                 # Streamlit application
├── crop_model.pkl         # Trained crop recommendation model
├── fertilizer_model.pkl   # Trained fertilizer recommendation model
├── scaler.pkl             # StandardScaler used for fertilizer model
├── requirements.txt       # Dependencies
└── README.md              # Project documentation


👨‍💻 Developed By

Parth Bhegade
BE Artificial Intelligence & Machine Learning
PES Modern College of Engineering, Pune
# FAA Bird Strike Incident Prediction and Analysis

This project predicts **aircraft damage level** and **repair cost** from FAA bird strike incident data, and uses **RAG (Retrieval-Augmented Generation)** to provide historical context and AI-powered explanations.  
A **Streamlit** app ties everything together with a user-friendly interface for both aviation safety experts and enthusiasts.

---

## ✨ Features

- **Classification (Random Forest)** → Predicts id damage happens or not

- **Regression (Gradient Boosting Regressor)** → Estimates **repair cost** in USD.

- **RAG + LLM** → Retrieves similar past incidents and generates **human-readable explanations**.

- **Interactive Streamlit App** → Enter incident details and get predictions & explanations instantly.

- **Deployment Ready** → Fully Dockerized and AWS-ready.

---

## 📊 Dataset

- **Source:** [FAA Wildlife Strike Database](https://wildlife.faa.gov/home)
- **Example fields:**
  - `INCIDENT_DATE`
  - `AIRPORT`
  - `AIRCRAFT`
  - `SPEED`
  - `PHASE_OF_FLIGHT`
  - `DAMAGE_LEVEL`
  - `COST_REPAIRS`

---

## 📓 Notebooks

### **EDA_Randomforest.ipynb**
- Cleans and preprocesses the dataset.
- Performs **Exploratory Data Analysis (EDA)** with visualizations.
- Trains a **Random Forest Classifier** to predict `DAMAGE_LEVEL`.
- Saves trained model as `models/rf_damage_predictor.pkl`.

### **Rag_Regress.ipynb**
- Trains a **Gradient Boosting Regressor** for predicting `COST_REPAIRS` as models/gbr_repair_cost_model.pkl
- Implements **RAG pipeline** using LangChain + FAISS.
- Stores incident records as embeddings in `vectorstore/`.

---

## 🖥 Streamlit App (`app.py`)

- Loads the **classification** and **regression** models.
- Loads **FAISS** vector store.
- Accepts user input for incident features (date, aircraft type, speed, phase of flight, etc.).
- Predicts **damage level** & **repair cost**.
- Retrieves and summarizes similar historical incidents.
- Displays results with an **LLM-generated explanation**.

---

## ⚙️ Installation

```bash
# Clone repo
git clone https://github.com/Arsalancr7/faa-bird-strike-prediction.git
cd faa-bird-strike-prediction

# (Optional) Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

---

## 🖥 Streamlit App (`app.py`)

- Loads the **classification** and **regression** models.
- Loads **FAISS** vector store.
- Accepts user input for incident features (date, aircraft type, speed, phase of flight, etc.).
- Predicts **damage level** & **repair cost**.
- Retrieves and summarizes similar historical incidents.
- Displays results with an **LLM-generated explanation**.

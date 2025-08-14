🚀 Drill String Failure Prediction System

An end-to-end predictive maintenance solution for drilling operations, built using Machine Learning + Deep Learning.
The system predicts the probability of drill string failure based on operational, mechanical, and environmental parameters — and provides actionable recommendations to reduce risk.

📌 Features

Data Preprocessing & Leakage Handling

Removed leaky features such as post-operation stress values & human-assigned risk levels.

Feature scaling & categorical encoding.

Model Development

Multiple ML approaches tested (Random Forest, Neural Networks).

Keras Tuner used for hyperparameter optimization.

Experimented with class balancing (SMOTE, class weights).

Interactive Web Application

Built with Streamlit for instant predictions.

Inputs grouped into sections (Material, Dimensional, Mechanical parameters).

Displays probability, prediction, and recommendations.

Generates downloadable prediction reports (.csv) for logging and analysis.

Explainability & Recommendations

Identifies key risk factors like high vibration levels or RPM settings.

Suggests operational adjustments to minimize failure risk.

🛠️ Tech Stack

Python 3.11+

Pandas, NumPy, Scikit-learn – Data processing & feature engineering

TensorFlow/Keras – Deep learning model training

Keras Tuner – Hyperparameter search

Streamlit – Web app UI

Matplotlib/Seaborn – Visualizations

📊 Dataset

The dataset contains operational, material, and mechanical parameters from drilling operations, including:

Depth, Weight on Bit (WOB)

Torque, RPM, Mud Weight

Vibration Level, Formation Type, Bit Type

Stress values (removed if leaky)

⚠️ Note: Dataset quality was a limitation; the focus of this project was pipeline development and deployment, not final accuracy.

📈 Model Performance

Due to dataset imbalance & noise, model performance varied — but the project demonstrates:

✅ Correct handling of train/test splits and leakage prevention
✅ Usage of hyperparameter tuning to improve results
✅ Deployment-ready Streamlit interface

🚀 Running the Project
# Clone the repo
git clone https://github.com/your-username/drill-failure-prediction.git
cd drill-failure-prediction

Run the Streamlit app
streamlit run app.py

📌 Example Prediction

Input Parameters:

Depth: 1695 m

WOB: 9.97 ton

RPM: 146

Torque: 15.61 kNm

Vibration Level: 5.0

Prediction: Failure (Probability: 99.8%)
Recommendation: Reduce RPM, inspect BHA vibration dampeners.

📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

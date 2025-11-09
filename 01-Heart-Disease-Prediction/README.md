# Heart Disease Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-Classification-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📋 Project Overview

A comprehensive machine learning project that predicts heart disease presence in patients using clinical features. This project compares multiple classification algorithms and achieves **94.17% AUC** with Support Vector Machine (SVM).

**Context:** Cardiovascular diseases are the leading cause of death globally, claiming approximately 17.9 million lives annually (31% of all deaths worldwide). Early detection through machine learning can significantly improve patient outcomes.

## 🎯 Key Results

| Model | Accuracy | AUC Score |
|-------|----------|-----------|
| **SVM (Best)** | **87.50%** | **94.17%** |
| Random Forest | 87.50% | 93.65% |
| Gradient Boosting | 86.41% | 92.84% |
| Logistic Regression | 85.33% | 92.68% |

## 🛠️ Technologies Used

- **Python 3.8+**
- **Libraries:**
  - NumPy & Pandas (Data manipulation)
  - Scikit-learn (ML models & metrics)
  - Matplotlib & Seaborn (Visualization)

## 📊 Dataset

- **Source:** [Kaggle Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Features:** 11 clinical features including age, blood pressure, cholesterol, etc.
- **Target:** Binary classification (Heart Disease: Yes/No)

## 🔍 Key Features

### 1. Comprehensive EDA
- Distribution analysis of target variable
- Numerical features visualization (Age, BP, Cholesterol, MaxHR, Oldpeak)
- Categorical features analysis (Sex, ChestPainType, RestingECG, etc.)
- Feature correlation studies

### 2. Data Preprocessing
- Missing value handling
- One-hot encoding for categorical variables
- StandardScaler normalization
- 80/20 train-test split

### 3. Model Comparison
- 4 different classification algorithms tested
- Multiple evaluation metrics (Accuracy, AUC, Precision, Recall, F1-Score)
- Confusion matrix analysis
- ROC curve visualization

### 4. Professional Visualizations
- Target distribution pie charts
- Feature distributions histograms
- Confusion matrices heatmaps
- ROC curves with AUC scores
- Model performance comparison charts

## 📁 Project Structure

```
01-Heart-Disease-Prediction/
├── Heart_Disease_Prediction.ipynb  # Main notebook
├── heart.csv                        # Dataset
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

## 🚀 How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook Heart_Disease_Prediction.ipynb
   ```

## 📈 Results Visualization

The project includes:
- **Confusion Matrix:** Shows SVM correctly predicted 87% of cases
- **ROC Curve:** Demonstrates excellent model discrimination (AUC = 0.9417)
- **Comparative Analysis:** Bar charts comparing all models

## 💡 Key Insights

1. **SVM performs best** with kernel='rbf' achieving the highest AUC score
2. **Random Forest** tied for accuracy but slightly lower AUC
3. **All models** performed well (>85% accuracy), indicating good feature quality
4. **Age, Chest Pain Type, and ST_Slope** showed strong correlation with heart disease

## 👨‍💻 Author

**Gabriel Sultan**  
Engineering Student - Data Science & AI  
[GitHub](https://github.com/GabrielSultan) | [LinkedIn](https://www.linkedin.com/in/gabriel-sultan)

## 📝 License

This project is for educational and portfolio purposes.

---

⭐ If you found this project helpful, please consider giving it a star!


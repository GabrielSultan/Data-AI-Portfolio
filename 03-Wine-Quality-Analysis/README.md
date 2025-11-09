# Wine Quality Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-Regression%20%26%20Classification-green.svg)
![Sklearn](https://img.shields.io/badge/Sklearn-Pipeline-orange.svg)

## 📋 Project Overview

A comprehensive machine learning project analyzing white wine quality using physicochemical properties. The project demonstrates both **regression** and **classification** approaches, with emphasis on proper ML pipelines and model comparison.

## 🎯 Key Results

### Regression Task (Quality Score 0-10)
| Configuration | MSE | R² Score |
|---------------|-----|----------|
| StandardScaler + 0.2 test | 0.6207 | 0.2587 |
| MinMaxScaler + 0.2 test | 0.6207 | 0.2587 |

### Binary Classification (Good/Bad Wine)
| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| **Naive Bayes** | **94.76%** | **0.8144** |
| Logistic Regression | 96.46% | 0.7675 |

⚠️ **Note:** High accuracy for Logistic Regression is misleading due to imbalanced dataset (96% positive class).

## 🍷 Dataset

- **Source:** [UCI ML Repository - Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Type:** White Wine (Vinho Verde, Portugal)
- **Samples:** 4,898 instances
- **Features:** 11 physicochemical properties
  - Fixed acidity
  - Volatile acidity
  - Citric acid
  - Residual sugar
  - Chlorides
  - Free sulfur dioxide
  - Total sulfur dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
- **Target:** Quality score (0-10) or Binary (Good/Bad)

## 🛠️ Technologies Used

- **Python 3.8+**
- **Libraries:**
  - Pandas & NumPy (Data manipulation)
  - Scikit-learn (ML pipelines & models)
  - Matplotlib (Visualization)

## 🔍 Project Highlights

### 1. Exploratory Data Analysis
```python
# Clean dataset - no missing values
wine_data.isnull().sum()  # All zeros ✓
```

### 2. Regression Analysis

**Approach:**
- Predicting continuous quality scores (1-10)
- Tested StandardScaler vs MinMaxScaler
- Evaluated different train/test splits (0.1, 0.2)

**Key Finding:** Scalers don't significantly impact linear regression performance.

**Model Formula:**
```
Quality = 5.89 + 0.038*fixed_acidity - 0.181*volatile_acidity 
          + 0.001*citric_acid + 0.391*residual_sugar 
          - 0.008*chlorides + ... + 0.263*alcohol
```

### 3. Binary Classification

**Approach:**
- Converting to binary problem: Quality ≥ 5 = Good (1), < 5 = Bad (0)
- Addressing class imbalance (96.27% positive class)
- Sklearn Pipeline for proper preprocessing
- Stratified train-test split

**Pipeline Structure:**
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GaussianNB())
])
```

### 4. Model Comparison

**Evaluation Metrics:**
- Accuracy
- Precision & Recall
- F1-Score
- AUC-ROC (most reliable for imbalanced data)
- Confusion Matrix

**Winner:** Naive Bayes with 81.44% AUC-ROC
- Better handles imbalanced data
- More reliable predictions
- Lower accuracy but higher AUC (more meaningful metric)

## 📁 Project Structure

```
03-Wine-Quality-Analysis/
├── Wine_Quality_Analysis.ipynb  # Main notebook
├── winequality-white.csv         # Dataset
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Jupyter notebook:
   ```bash
   jupyter notebook Wine_Quality_Analysis.ipynb
   ```

## 📊 Key Visualizations

1. **Actual vs Predicted:** Scatter plot showing model predictions
2. **Confusion Matrices:** For both Logistic Regression and Naive Bayes
3. **Feature Importance:** Through coefficient analysis

## 💡 Key Insights

### Technical Insights:
1. **Normalization Impact:** Y-normalization doesn't affect linear regression (scale-invariant)
2. **Scaler Choice:** StandardScaler and MinMaxScaler perform identically for this dataset
3. **Class Imbalance:** 96% positive class makes accuracy misleading
4. **Metric Selection:** AUC-ROC is more reliable than accuracy for imbalanced data

### Domain Insights:
1. **Alcohol** is the most important positive predictor (+0.263)
2. **Volatile acidity** is the strongest negative predictor (-0.181)
3. **Residual sugar** has moderate positive impact (+0.391)

### Limitations:
- R² of 0.26 indicates linear models may not capture full complexity
- Consider non-linear models (Random Forest, XGBoost) for better performance
- More feature engineering could improve results

## 🎓 What This Project Demonstrates

1. **Sklearn Pipelines:** Best practices for ML workflows
2. **Regression vs Classification:** Understanding when to use each
3. **Imbalanced Data Handling:** Proper metric selection
4. **Feature Engineering:** Transforming continuous to binary target
5. **Model Comparison:** Systematic evaluation approach
6. **Critical Analysis:** Understanding limitations and misleading metrics

## 🔮 Future Improvements

- Implement ensemble methods (Random Forest, XGBoost)
- Feature engineering (interaction terms, polynomial features)
- Handle class imbalance (SMOTE, class weights)
- Cross-validation for robust evaluation
- Compare with red wine dataset
- Deploy as web app for wine quality prediction

## 👨‍💻 Author

**Gabriel Sultan**  
Engineering Student - Data Science & AI  
[GitHub](https://github.com/GabrielSultan) | [LinkedIn](https://www.linkedin.com/in/gabriel-sultan)

## 📝 License

This project is for educational and portfolio purposes.

## 🍾 References

- P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, Elsevier, 47(4):547-553, 2009.

---

⭐ If you found this analysis useful, please give it a star!


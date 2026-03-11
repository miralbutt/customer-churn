# 📊 Customer Churn Prediction

**Author:** Miral Butt | Data Analyst | MPhil Statistics, LCWU Lahore

---

## 🎯 Project Overview

Customer churn — when customers stop using a service — costs businesses millions every year. This project uses **Machine Learning** to predict which customers are at risk of churning, so businesses can take action before it's too late.

---

## 📁 Project Structure

```
customer-churn/
│
├── Customer_Churn_Prediction.ipynb   # Main analysis notebook
├── eda_plots.png                     # Exploratory Data Analysis charts
├── model_results.png                 # Model performance charts
└── README.md                         # Project documentation
```

---

## 🔍 What's Inside

### 1. Exploratory Data Analysis
- Churn rate distribution
- Churn by Contract Type
- Tenure vs Churn patterns
- Monthly Charges analysis

### 2. Key Findings
| Factor | Impact on Churn |
|---|---|
| Month-to-Month Contract | Highest churn risk |
| Tenure < 12 months | 25% higher churn probability |
| Monthly Charges > $80 | Significantly more likely to churn |
| Support Calls > 3 | Strong churn indicator |

### 3. Models Used
| Model | Accuracy |
|---|---|
| Logistic Regression | ~78% |
| Random Forest | ~83% |

### 4. Business Recommendations
- 🔴 Offer discounts to Month-to-Month customers to switch to annual plans
- 🔴 Create retention program for customers in their first 12 months
- 🟡 Proactively contact customers with 3+ support calls
- 🟢 Use ML model to score all customers monthly and flag at-risk ones

---

## 🛠️ Tools & Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat)

- **Python** — Data analysis & modeling
- **Pandas & NumPy** — Data manipulation
- **Matplotlib & Seaborn** — Visualization
- **Scikit-learn** — Machine Learning (Logistic Regression, Random Forest)

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/miralbutt/customer-churn.git

# Install dependencies
pip install pandas scikit-learn matplotlib seaborn jupyter

# Open notebook
jupyter notebook Customer_Churn_Prediction.ipynb
```

---

## 📬 Contact

**Miral Butt**
📧 miralb246@gmail.com
🔗 [GitHub Profile](https://github.com/miralbutt)

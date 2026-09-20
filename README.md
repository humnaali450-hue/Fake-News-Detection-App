# 📰 Fake News Detection App

A Machine Learning-based web application that classifies news articles as **REAL** or **FAKE** using Natural Language Processing techniques.

The project uses **TF-IDF Vectorization** for text representation and compares multiple machine learning algorithms including **Logistic Regression, Linear SVC, Naive Bayes, and Random Forest**. The trained model is deployed through an interactive **Streamlit web application**.

---

## 🚀 Features

### 🧹 Text Preprocessing

The application cleans raw news text before prediction by:

* Converting text to lowercase
* Removing special characters
* Removing punctuation
* Removing unnecessary spaces
* Removing English stopwords
* Preparing text for machine learning

### 🤖 Machine Learning Models

The project compares multiple classification algorithms:

* Logistic Regression
* Linear Support Vector Classifier (LinearSVC)
* Multinomial Naive Bayes
* Random Forest Classifier

The best-performing model can be selected and saved for deployment.

### 📊 Model Evaluation

Models are evaluated using several classification metrics:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix
* ROC Curve
* ROC-AUC Score

### 📈 Prediction Confidence

The web application displays the predicted news category along with a confidence score representing how certain the model is about its prediction.

### 🌐 Interactive Streamlit Application

Users can paste a news article into the application and receive an instant prediction indicating whether the article is:

**✅ REAL NEWS**

or

**🚨 FAKE NEWS**

### 📉 Data Visualization

The project includes several useful visualizations:

* Word Cloud for Fake News
* Word Cloud for Real News
* Confusion Matrix
* ROC Curve
* News Article Length Distribution
* Class Distribution

### ⚙️ Hyperparameter Tuning

The project can use:

* Cross Validation
* GridSearchCV

to improve model performance and identify better hyperparameter combinations.

---

# 🧠 Project Workflow

The complete machine learning pipeline follows these steps:

```text
News Dataset
      ↓
Data Cleaning
      ↓
Text Preprocessing
      ↓
Exploratory Data Analysis
      ↓
TF-IDF Vectorization
      ↓
Machine Learning Models
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
Model Serialization
      ↓
Streamlit Deployment
```
---

# 📂 Dataset

The project uses two news datasets:

### Fake.csv

Contains fake news articles.

```text
Label = 0
```

### True.csv

Contains real news articles.

```text
Label = 1
```

The datasets are combined and processed before model training.

---

# 🔬 Text Preprocessing

A text cleaning function such as `clean_text()` is used to prepare news articles before feature extraction.

Typical preprocessing steps include:

```python
def clean_text(text):
    text = text.lower()
    # remove punctuation
    # remove special characters
    # remove stopwords
    # normalize spaces
    return text
```

The cleaned text is then converted into numerical features using **TF-IDF Vectorization**.

---

# 🔢 TF-IDF Vectorization

Machine learning algorithms cannot directly process raw text.

Therefore, the project uses:

```python
TfidfVectorizer
```

TF-IDF converts news articles into numerical feature vectors based on the importance of words within documents and across the entire dataset.

---

# 🤖 Model Training

The project trains and compares multiple machine learning models:

```text
Logistic Regression
Naive Bayes
Random Forest
Linear SVC
```

General training process:

```text
Preprocessed Text
        ↓
TF-IDF Features
        ↓
Train-Test Split
        ↓
Model Training
        ↓
Prediction
        ↓
Evaluation
```

---

# 💾 Model Saving

After selecting the preferred model, the trained model and TF-IDF vectorizer can be saved using Python's `pickle` module.

Example:

```python
import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
```

These files can then be loaded by the Streamlit application without retraining the model every time.

---

# ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

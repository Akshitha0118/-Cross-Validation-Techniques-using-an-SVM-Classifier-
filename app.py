# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform


# ==========================================
# 2. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="SVM Classification Dashboard",
    layout="wide"
)


# ==========================================
# 3. CUSTOM CSS
# ==========================================
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #38bdf8;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 30px;
}
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    color: black;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.2);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. TITLE
# ==========================================
st.markdown('<div class="title">SVM Classification Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Social Network Ads Dataset</div>', unsafe_allow_html=True)


# ==========================================
# 5. LOAD DATA
# ==========================================
dataset = pd.read_csv(r"C:\Users\Admin\Desktop\cross validation svc\Social_Network_Ads.csv")

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values


# ==========================================
# 6. TRAIN TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)


# ==========================================
# 7. FEATURE SCALING
# ==========================================
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ==========================================
# 8. MODEL TRAINING
# ==========================================
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)


# ==========================================
# 9. PREDICTION
# ==========================================
y_pred = classifier.predict(X_test)


# ==========================================
# 10. METRICS
# ==========================================
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
bias = classifier.score(X_train, y_train)
variance = classifier.score(X_test, y_test)


# ==========================================
# 11. DISPLAY METRICS
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="card">
        <h3>Accuracy</h3>
        <h2>{accuracy:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <h3>Training Accuracy (Bias)</h3>
        <h2>{bias:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <h3>Testing Accuracy (Variance)</h3>
        <h2>{variance:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 12. CONFUSION MATRIX
# ==========================================
st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
ax.matshow(cm)
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, val, ha='center', va='center')

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)


# ==========================================
# 13. CROSS VALIDATION
# ==========================================
accuracies = cross_val_score(
    estimator=classifier,
    X=X_train,
    y=y_train,
    cv=8
)

st.markdown(f"""
<div class="card">
    <h3>Cross Validation Accuracy</h3>
    <h2>{accuracies.mean() * 100:.2f} %</h2>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 14. GRID SEARCH CV
# ==========================================
parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'kernel': ['rbf']}
]

grid_search = GridSearchCV(
    estimator=classifier,
    param_grid=parameters,
    scoring='accuracy',
    cv=10,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

st.subheader("Grid Search Results")
st.write("Best Accuracy:", grid_search.best_score_)
st.write("Best Parameters:", grid_search.best_params_)


# ==========================================
# 15. RANDOM SEARCH CV
# ==========================================
parameters_random = {
    'C': uniform(1, 1000),
    'kernel': ['linear', 'rbf'],
    'gamma': uniform(0.01, 1)
}

random_search = RandomizedSearchCV(
    estimator=classifier,
    param_distributions=parameters_random,
    n_iter=50,
    scoring='accuracy',
    cv=10,
    random_state=0,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

st.subheader("Randomized Search Results")
st.write("Best Accuracy:", random_search.best_score_)
st.write("Best Parameters:", random_search.best_params_)

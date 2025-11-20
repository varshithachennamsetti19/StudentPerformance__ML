import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import time as t
import sklearn.utils as u
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.metrics as m
import sklearn.linear_model as lm
import sklearn.neural_network as nn
from sklearn.model_selection import train_test_split
import numpy as np
import warnings as w
w.filterwarnings('ignore')

# Load dataset
data = pd.read_csv("AI-Data.csv")

# ---- NEW FEATURE: Correlation Heatmap ----
plt.figure(figsize=(12, 8))
sb.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
# -----------------------------------------

# MENU FOR GRAPHS
ch = 0
while(ch != 10):
    print("1.Marks Class Count Graph\t2.Marks Class Semester-wise Graph\n3.Marks Class Gender-wise Graph\t4.Marks Class Nationality-wise Graph\n5.Marks Class Grade-wise Graph\t6.Marks Class Section-wise Graph\n7.Marks Class Topic-wise Graph\t8.Marks Class Stage-wise Graph\n9.Marks Class Absent Days-wise\t10.No Graph\n")
    ch = int(input("Enter Choice: "))

    if ch == 1:
        sb.countplot(x='Class', data=data, order=['L', 'M', 'H'])
        plt.show()

    elif ch == 2:
        sb.countplot(x='Semester', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.show()

    elif ch == 3:
        sb.countplot(x='gender', hue='Class', data=data, order=['M', 'F'], hue_order=['L', 'M', 'H'])
        plt.show()

    elif ch == 4:
        sb.countplot(x='NationalITy', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.show()

    elif ch == 5:
        sb.countplot(x='GradeID', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.show()

    elif ch == 6:
        sb.countplot(x='SectionID', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.show()

    elif ch == 7:
        sb.countplot(x='Topic', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.show()

    elif ch == 8:
        sb.countplot(x='StageID', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.show()

    elif ch == 9:
        sb.countplot(x='StudentAbsenceDays', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.show()

print("Exiting graphs...\n")

# DROP COLUMNS (ONLY NUMERIC FEATURES LEFT)
data = data.drop(["gender", "StageID", "GradeID", "NationalITy", "PlaceofBirth",
                  "SectionID", "Topic", "Semester", "Relation", "ParentschoolSatisfaction",
                  "ParentAnsweringSurvey", "AnnouncementsView"], axis=1)

# Encode non-numerical values
for column in data.columns:
    if data[column].dtype == object:
        le = pp.LabelEncoder()
        data[column] = le.fit_transform(data[column])

# FEATURES & LABELS (Important → ONLY FIRST 4 columns used)
feats = data.values[:, 0:4]
lbls = data.values[:, 4]

# ---------- CORRECT TRAIN-TEST SPLIT -----------
X_train, X_test, y_train, y_test = train_test_split(
    feats, lbls, test_size=0.3, random_state=42
)
# ------------------------------------------------

# MODELS
modelD = tr.DecisionTreeClassifier()
modelR = es.RandomForestClassifier()
modelP = lm.Perceptron()
modelL = lm.LogisticRegression()
modelN = nn.MLPClassifier(activation="logistic")

# TRAINING
modelD.fit(X_train, y_train)
modelR.fit(X_train, y_train)
modelP.fit(X_train, y_train)
modelL.fit(X_train, y_train)
modelN.fit(X_train, y_train)

# PREDICTIONS
predD = modelD.predict(X_test)
predR = modelR.predict(X_test)
predP = modelP.predict(X_test)
predL = modelL.predict(X_test)
predN = modelN.predict(X_test)

print("\nDecision Tree Report:\n", m.classification_report(y_test, predD))
print("\nRandom Forest Report:\n", m.classification_report(y_test, predR))
print("\nPerceptron Report:\n", m.classification_report(y_test, predP))
print("\nLogistic Regression Report:\n", m.classification_report(y_test, predL))
print("\nNeural Network Report:\n", m.classification_report(y_test, predN))

# ASK USER FOR NEW INPUT
choice = input("\nDo you want to test an input? (y/n): ")

if choice.lower() == "y":
    rai = int(input("Enter raised hands: "))
    res = int(input("Enter Visited Resources: "))
    dis = int(input("Enter No. of Discussions: "))
    absc = int(input("Enter Absences (1=Under-7 , 0=Above-7): "))

    arr = np.array([rai, res, dis, absc]).reshape(1, -1)

    pD = modelD.predict(arr)[0]
    pR = modelR.predict(arr)[0]
    pP = modelP.predict(arr)[0]
    pL = modelL.predict(arr)[0]
    pN = modelN.predict(arr)[0]

    mapping = {0: "H", 1: "M", 2: "L"}

    print("\nUsing Decision Tree:", mapping[pD])
    print("Using Random Forest:", mapping[pR])
    print("Using Perceptron:", mapping[pP])
    print("Using Logistic Regression:", mapping[pL])
    print("Using Neural Network:", mapping[pN])

print("\nExiting...")

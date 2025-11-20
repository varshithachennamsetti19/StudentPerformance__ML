This project predicts student performance levels (Low / Medium / High) based on their activity and engagement in an online learning environment. Multiple machine-learning algorithms are used and compared to find the most accurate prediction.

The dataset includes features such as: raisedhands VisITedResources AnnouncementsView Discussion gender, Nationality, SectionID, GradeID, etc.

This project gives predictions using models like: Decision Tree Random Forest Logistic Regression Perceptron Neural Network

Each model outputs a prediction (Example: L, M, H).

Features ✔ Data loading and preprocessing ✔ Label encoding for categorical features ✔ Train–test splitting ✔ Training multiple ML models ✔ Evaluating accuracy ✔ Predicting the class for new student data ✔ Comparison of all models

Machine Learning Models Used Decision Tree Classifier

Random Forest Classifier

Logistic Regression

Perceptron

Neural Network (MLPClassifier)

Each model gives a prediction such as: Using Decision Tree: M Using Random Forest: M Using Perceptron: H Using Logistic Regression: M Using Neural Network: M

Project Structure project/ │── Project.py # Main ML code │── dataset.csv # Input dataset │── requirements.txt # Python dependencies │── README.md # Documentation

Installation

Create virtual environment (optional) python -m venv venv

Activate it Windows: venv\Scripts\activate

Install dependencies pip install -r requirements.txt

Run the command: python Project.py

After running, the program will:

Load data
Train models
Print model predictions
Show accuracy of each model

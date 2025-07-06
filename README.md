# 🩺 Diabetes Prediction using SVM
This project uses machine learning (SVM classifier) to predict whether a person is diabetic or not based on the PIMA Diabetes dataset.

# 📊 Dataset Description
The dataset diabetes.csv is from the PIMA Indian Diabetes dataset and contains the following features:

Pregnancies: Number of times pregnant

Glucose: Plasma glucose concentration

BloodPressure: Diastolic blood pressure (mm Hg)

SkinThickness: Triceps skinfold thickness (mm)

Insulin: 2-Hour serum insulin (mu U/ml)

BMI: Body mass index (weight in kg/(height in m)^2)

DiabetesPedigreeFunction: A function that scores likelihood of diabetes based on family history

Age: Age of the person

Outcome: 1 for diabetic, 0 for non-diabetic

# 🚀 How to Run
Requirements
Python 3.x

pandas, numpy, scikit-learn

Install Dependencies

pip install pandas numpy scikit-learn
Run the Project
Make sure diabetes.csv is in the correct path. Then execute:
python Prediction.py

The script will:

Preprocess and standardize the data

Train a Support Vector Machine (SVM) model

Evaluate accuracy on both training and test data

Predict diabetes based on a sample input

# 🧪 Sample Prediction
Example input tested in Prediction.py:

input_data = (4,110,92,0,0,37.6,0.191,30)
The model will predict and print:

The person is diabetic
or
The person is not diabetic

# 📈 Accuracy
The model uses an 80-20 train-test split and prints accuracy scores for both.

# 📌 Notes
Make sure to adjust the dataset path in Prediction.py if not running from the same location:

pd.read_csv('diabetes.csv')
You can modify input_data at the end of the script to test different cases.


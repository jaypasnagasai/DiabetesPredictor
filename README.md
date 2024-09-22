# Diabetes Predictor

[Original Project](https://colab.research.google.com/drive/1oxnhMTlomJ4HVhPuowpPFyMt1mwuOuQo?usp=sharing#scrollTo=mmJ22qhVvNwj)

## Overview
This notebook presents a machine learning model to predict whether a person is diabetic based on various health-related features such as glucose levels, BMI, and insulin. The prediction is based on a dataset and includes data preprocessing steps, model training, evaluation, and a prediction based on user input.

---

## Dataset Description
The dataset used in this project is composed of the following features:
- **Pregnancies**: Number of times the individual has been pregnant.
- **Glucose**: Plasma glucose concentration.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skinfold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg/(height in m)^2).
- **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history.
- **Age**: Age of the individual.
- **Outcome**: Binary value indicating whether the individual has diabetes (1) or not (0).

---

## Steps in the Notebook

### 1. Loading the Dataset
The dataset is loaded using the `pandas` library and displayed to provide a glimpse of its structure and the data it contains.

### 2. Data Preprocessing
The dataset is divided into features (X) and labels (Y), where:
- `X` contains all the features except for the `Outcome`.
- `Y` contains the `Outcome`, which is the label indicating if the person has diabetes.

### 3. Model Selection
The model used in the notebook is a machine learning classifier. The following steps are performed:
- **Scaling the Data**: The features are standardized using `StandardScaler` to ensure all values have the same scale for better performance of the model.
- **Training the Model**: The model is trained using a dataset split into training and testing sets. The classifier used is not explicitly stated, but the notebook includes steps for training and making predictions.

### 4. Making Predictions
The model allows for manual input of a data instance to make predictions. For example:
- A sample input tuple is provided (`InputData = (5,166,72,19,175,25.8,0.587,51)`), representing the feature values for a single individual.
- This data is reshaped and passed through the standardization process using the scaler.
- The model then predicts whether the individual is diabetic or not.

### 5. Model Output
- If the prediction result is `0`, the person is not diabetic.
- If the prediction result is `1`, the person is diabetic.

---

## Example Code for Prediction

```python
# Example Input Data
InputData = (5,166,72,19,175,25.8,0.587,51)

# Convert Input Data to NumPy Array
NumpyArray = np.asarray(InputData)

# Reshape the Array
ReshapeArray = NumpyArray.reshape(1, -1)

# Standardize the Input Data
StdData = scaler.transform(ReshapeArray)

# Make Prediction
P = classifier.predict(StdData)

# Output the Result
if (P[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')

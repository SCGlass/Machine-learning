# Importing relative packages pandas for the dataframe, joblib to import model
# and classification report to see model results
import pandas as pd
import joblib
from sklearn.metrics import classification_report

# importing the 100 test data
test_data =  "data\Test_data_disease.csv"
df_test = pd.read_csv(test_data)

# dropping the cardio column from the data frame as this is what we want to predict
X,y = df_test.drop("cardio", axis = "columns"), df_test["cardio"]

# importing model
model = joblib.load("Labb/voting_clf_model.pkl")

# getting predictions from the model
y_pred = model.predict(X)

# getting the predicted probabilities from the model
proba = model.predict_proba(X)

# cresting 2 variables for the probability by loc with the array from proba
proba_class_0 = proba[:,0]
proba_class_1 = proba[:,1]

# creating a dictionary from the results to be able to make a data frame, with titles
data = {"probability class 0" : proba_class_0,
        "probability class 1" : proba_class_1,
        "prediction" : y_pred}

# creating a data frame
df_predictions = pd.DataFrame(data)

# saving results data frame as a csv file
df_predictions.to_csv("data\predictions.csv", index=False)

# printing out accuracy score of the model and a classification report to see the results
print(f"Accuracy score:  {model.score(X,y)}" )
print(classification_report(y_pred, y))

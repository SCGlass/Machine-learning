import pandas as pd
import joblib
from sklearn.metrics import classification_report

test_data =  "data\Train_data_disease.csv"
df_test = pd.read_csv(test_data)

X,y = df_test.drop("cardio", axis = "columns"), df_test["cardio"]

model = joblib.load("Labb/voting_clf_model.pkl")

y_pred = model.predict(X)

proba = model.predict_proba(X)

proba_class_0 = proba[:,0]
proba_class_1 = proba[:,1]

data = {"probability class 0" : proba_class_0,
        "probability class 1" : proba_class_1,
        "prediction" : y_pred}

df_predictions = pd.DataFrame(data)

df_predictions.to_csv("data\predictions.csv", index=False)

print(f"Accuracy score:  {model.score(X,y)}" )
print(classification_report(y_pred, y))

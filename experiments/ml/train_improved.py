import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mlflow.set_experiment('ml-experiments')

X,y = load_wine(return_X_y= True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run(run_name="log_reg_improved"):
    model = LogisticRegression(C=0.3, max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("model", "logistic_regression")
    mlflow.log_param("C", 0.3)
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, name="Logistic_Regression_improved_model")


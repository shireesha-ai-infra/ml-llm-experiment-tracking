import mlflow
import torch
import torch.nn as nn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mlflow.set_experiment("nn_experiments")

X, y = load_wine(return_X_y=True)

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

model = nn.Sequential(
    nn.Linear(X.shape[1], 32),
    nn.ReLU(),
    nn.Linear(32,3)
)

criterian = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

with mlflow.start_run(run_name="nn_baseline"):
    for epoch in range(30):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterian(outputs, y_train)
        loss.backward()
        optimizer.step()

        mlflow.log_metric("loss", loss.item(), step= epoch)
    mlflow.log_param("epochs", 30)
    mlflow.log_param("hidden_units", 32)
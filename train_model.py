import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

mlflow.set_tracking_uri('http://localhost:5000')

def train_log_model(n_estimators, max_depth):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    with mlflow.start_run():
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)

        model = RandomForestClassifier()
        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)

        run_id = mlflow.active_run().info.run_id

        mlflow.sklearn.log_model(model, "fin-app")
        model_uri = f"runs:/{run_id}/fin-app"

        registered_model = mlflow.register_model(model_uri, "FinApp")

        client = MlflowClient()
        client.set_model_version_tag(
            name='FinApp',
            version=registered_model.version,
            key='accuracy',
            value=str(round(accuracy, 3))
        )
def promote_best_model(model_name):
    client = MlflowClient()
    best_accuracy = 0
    best_version =None
    for version in client.search_model_versions(f"name:{model_name}"):
        tmp_accuracy = version.tags.get("accuracy")
        if tmp_accuracy:
            tmp_accuracy = float(tmp_accuracy)
            if tmp_accuracy>best_accuracy:
                best_accuracy = tmp_accuracy
                best_version = version

    if best_version:
        client.transition_model_version_stage(
            name=best_version.name,
            version=best_version.version,
            stage="Production"
        )

if __name__=="__main__":
    train_log_model(50, 5)
    train_log_model(100, 80)
    train_log_model(400, None)

    promote_best_model('FinApp')





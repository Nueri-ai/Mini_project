from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Загрузка модели
my_model = joblib.load('fin_model.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['AtBat'],
                         data['Hits'],
                         data['Runs'],
                         data['Walks'],
                         data['Years'],
                         data['CRBIxCRuns'],
                         data['PutOuts'],
                         data['CRBI'],
                         data['CRuns'],
                         data['CAtBat'],
                         ]).reshape(1, -1)

    prediction = my_model.predict(features).tolist()
    return jsonify({'class': prediction[0]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

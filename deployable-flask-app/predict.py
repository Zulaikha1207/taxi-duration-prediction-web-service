import pickle
from flask import Flask, request, jsonify

with open('reports/xgboost.bin', 'rb') as f_in:
    (dv, model)= pickle.load(f_in)

def prepare_features(rides):
    features_list = []
    for ride in rides:
        features = {}
        features['PULocationID'] = ride['PULocationID']
        features['DOLocationID'] = ride['DOLocationID']
        features['trip_distance'] = ride['trip_distance']
        features_list.append(features)
    return features_list

def predict(features_list):
    X = dv.transform(features_list)
    preds = model.predict(X)
    return list(preds)

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    rides = request.get_json()

    features_list = prepare_features(rides)
    preds = predict(features_list)

    results = []
    for pred in preds:
        result = {
            'duration in minutes': float(pred)
        }
        results.append(result)

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

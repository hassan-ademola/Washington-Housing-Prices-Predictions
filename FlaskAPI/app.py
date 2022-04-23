import flask
from flask import Flask, jsonify, request
import json
import pandas as pd
import pickle

app = Flask(__name__)
app.config['DEBUG'] = True

def load_model():
    return pickle.load(open('models/model.pkl','rb'))

@app.route('/predict', methods=['GET'])
def predict():
    
    data = request.get_json()
    keys = data.keys()
    values = data.values()
    df = pd.DataFrame([values],columns=keys)
    X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
        'yr_built', 'yr_renovated', 'street', 'city', 'zip']]
        
    model = load_model()
    y = model.predict(X)
    response = json.dumps({'price': float(y)})
    return response, 200
    
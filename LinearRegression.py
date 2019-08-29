import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = []
    for x in request.form.values():
        features.append(float(x))
    final_features = [np.array(features)]
    predict = model.predict(final_features)

    return render_template('index.html', prediction_text=round(predict[0][0]))


if __name__ == "__main__":
    app.run(debug=True)

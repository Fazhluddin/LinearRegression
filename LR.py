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
    z = []
    features = []
    for x in request.form.values():
        z.append(x)
    a = [z[0], z[1], z[2]]
    for c in request.form.values():
        if c in a:
            features.append(float(c))
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    return render_template('index.html', prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True)

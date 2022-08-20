import werkzeug
from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd

model = pickle.load(open('svm_camelia_pickle_model_S6.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        filename = werkzeug.utils.secure_filename(file.filename)
        filestream = file
        filestream.seek(0)
        df = pd.read_csv(filestream, sep=",", header=None)
        amplitudes = df[1].values
        amplitudes = np.array(amplitudes)
        amplitudes = amplitudes.reshape(1, -1)
        result = model.predict(amplitudes)
        return jsonify({'Result': str(result)})

if __name__ == '__main__':
    app.run(port=4996)
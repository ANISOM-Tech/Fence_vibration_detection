# import the libraries
import json
import numpy as np
import pickle
from flask import Flask, render_template, jsonify, request


model = pickle.load(open('V_Train.sav', 'rb'))

app = Flask(__name__)
prediction = 'No-disturance'
no_movement = np.array([0])


@app.route('/model_predict')
def model_predict():
    return jsonify(result=prediction)


@app.route('/', methods=['POST'])
def get_feature():
    global prediction, no_movement
    if request.json:
        feat = json.loads(request.json)
        feat = np.array(feat)
        predict_value = model.predict(feat.reshape(1, 28))
        if predict_value == no_movement:
            prediction = 'No_disturbance'
        else:
            prediction = 'Disturbance'
        print(prediction)
    return 'received'


@app.route('/')
def index():
    return render_template('model_predict.html')


if __name__ == '__main__':
    app.debug = True
    app.run()

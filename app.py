from flask import Flask
from flask import request
#from mnist-mlops.utils import load
import numpy as np

app = flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    input_json = request.json
    image = input_json['image']
    print(image)
    return None


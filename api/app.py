from flask import Flask
from flask import request
from joblib import load
import numpy as np

app = Flask(__name__)

best_model_path = '/mnt/c/Users/Stuti Pathak/Desktop/IIT Jodhpur/Semester 3/MLOps/MNIST-MLOPS/models/SVM/tt_0.15_val_0.15_rescale_1_hyperp_0.001/model.joblib'

@app.route('/predict', methods=['POST'])
def predict():
    clf = load(best_model_path)
    input_json = request.json
    image = input_json['image']
    print(image)
    image = np.array(image).reshape(1, -1)
    predicted = clf.predict(image)
    return str(predicted[0])
    #return "<p>Image Obtained</p>"



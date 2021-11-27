from flask import Flask
from flask import request
from joblib import load
import numpy as np
import os

app = Flask(__name__)

#current_directory = os.getcwd()
#parent_directory = os.path.dirname(current_directory)
#svm_model_path = parent_directory + '/models/SVM/tt_0.15_val_0.15_rescale_1_hyperp_0.001/model.joblib'
#dt_model_path = parent_directory + '/models/DecisionTree/tt_0.15_val_0.15_rescale_1_hyperp_14/model.joblib'
current_directory = os.getcwd()
print(current_directory)
svm_model_path = current_directory + '/model.joblib'
dt_model_path = current_directory + '/model.joblib'



@app.route('/svm_predict', methods=['POST'])
def svm_predict():
    clf = load(svm_model_path)
    input_json = request.json
    image = input_json['image']
    print(image)
    image = np.array(image).reshape(1, -1)
    predicted = clf.predict(image)
    return str(predicted[0])

@app.route('/dt_predict', methods=['POST'])
def dt_predict():
    clf = load(dt_model_path)
    input_json = request.json
    image = input_json['image']
    print(image)
    image = np.array(image).reshape(1, -1)
    predicted = clf.predict(image)
    return str(predicted[0])
	
if __name__ == '__main__':
   app.run(host='0.0.0.0',debug=True)





import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

hmodel=pickle.load(open('hmodel.pkl','rb'))
dmodel=pickle.load(open('dmodel.pkl','rb'))
cmodel = pickle.load(open('cmodel.pkl', 'rb'))
symodel=pickle.load(open('symodel.pkl','rb'))

@app.route('/cvd')
def home0():
    return render_template('heart.html')

@app.route('/diabetes')
def home1():
    return render_template('diabetes.html')

@app.route('/cancer')
def home2():
    return render_template('cancer.html')

@app.route('/symptom')
def home3():
    return render_template('symptom.html')

@app.route('/predict_sympt',methods=['POST'])
def predict_sympt():
  
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = symodel.predict(final_features)

    output = prediction

    if output==0:
        return render_template('symptom.html', prediction_text="THE ABOVE SYMPTOMS SUGGEST YOU ARE SUFFERING FROM CARDIOVASCULAR DISEASE")
    if output==1:
        return render_template('symptom.html', prediction_text="THE ABOVE SYMPTOMS SUGGEST YOU ARE SUFFERING FROM DIABETES")
    if output==2:
        return render_template('symptom.html', prediction_text="THE ABOVE SYMPTOMS SUGGEST YOU ARE SUFFERING FROM CANCER")
    if output==3:
        return render_template('symptom.html', prediction_text="THE ABOVE SYMPTOMS SUGGEST YOU ARE SUFFERING FROM TUBERCULOSIS")

@app.route('/predict_heart',methods=['POST'])
def predict_heart():
  
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = hmodel.predict(final_features)

    output = round(prediction[0], 2)
    if output==1:
        return render_template('heart.html', prediction_text="YOU HAVE BEEN DETECTED FOR CARDIOVASCULAR DISEASE")
    else:
        return render_template('heart.html', prediction_text="YOU HAVE NOT BEEN DETECTED FOR CARDIOVASCULAR DISEASE")

@app.route('/predict_diabetes',methods=['POST'])
def predict_diabetes():
  
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = dmodel.predict(final_features)

    output = round(prediction[0], 2)

    if output==1:
        return render_template('diabetes.html', prediction_text='YOU ARE HAVEING DIABETES')
    else:
       return render_template('diabetes.html', prediction_text='YOU ARE NOT HAVING DIABETES')    

@app.route('/predict_cancer',methods=['POST'])
def predict_cancer():
  
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = cmodel.predict(final_features)

    output = round(prediction[0], 2)

    if output==1:
        return render_template('cancer.html', prediction_text="YOU MY BE SUFFERING FROM MALIGANT CANCER/TUMOR")
    else:
        return render_template('cancer.html', prediction_text="YOU MY BE SUFFERING FROM BENIGN CANCER/TUMOR")

if __name__ == "__main__":
    app.run(debug=True)

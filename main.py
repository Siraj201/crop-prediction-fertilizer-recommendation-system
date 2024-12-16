from flask import Flask, request, render_template
import pickle
import joblib
import pandas as pd


app = Flask(__name__)

#importing pickle files
model = pickle.load(open('models/fertilizer/classifier.pkl','rb'))
ferti = pickle.load(open('models/fertilizer/fertilizer.pkl','rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/crop')
def crop():
    return render_template('crop/index.html')

@app.route('/fertilizer')
def fertilizer():
    return render_template('fertilizer/index.html')


@app.route('/fertilizer-predict',methods=['POST'])
def predict():
    temp = request.form.get('temp')
    humi = request.form.get('humid')
    mois = request.form.get('mois')
    soil = request.form.get('soil')
    crop = request.form.get('crop')
    nitro = request.form.get('nitro')
    pota = request.form.get('pota')
    phosp = request.form.get('phos')
    input = [int(temp), int(humi), int(mois), int(soil), int(crop), int(nitro), int(pota), int(phosp)]

    res = ferti.classes_[model.predict([input])]

    return render_template('fertilizer/index.html',x = ('Predicted Fertilizer is {}'.format(res)))





@app.route('/crop-predict', methods=["POST"])
def brain():
    Nitrogen=float(request.form['Nitrogen'])
    Phosphorus=float(request.form['Phosphorus'])
    Potassium=float(request.form['Potassium'])
    Temperature=float(request.form['Temperature'])
    Humidity=float(request.form['Humidity'])
    Ph=float(request.form['ph'])
    Rainfall=float(request.form['Rainfall'])
     
    values=[Nitrogen,Phosphorus,Potassium,Temperature,Humidity,Ph,Rainfall]

    if Ph > 0 and Ph <= 14 and Temperature < 100 and Humidity > 0:
     joblib.load('cropp_app', 'r')
    model = joblib.load(open('cropp_app', 'rb'))
    arr = [values]
    probabilities = model.predict_proba(arr)[0]
    top_3_indices = probabilities.argsort()[-3:]
    recommended_crops = [(model.classes_[i], probabilities[i] * 100) for i in top_3_indices]
    recommended_crops.sort(key=lambda x: x[1], reverse=True)
    # print(acc)
    return render_template('crop/index.html', Predictions=str(recommended_crops))
    #else:
    return "Sorry...  Error in entered values in the form Please check the values and fill it again"

if __name__ == "__main__":
    app.run(debug=True)
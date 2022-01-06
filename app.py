from flask import Flask,render_template,jsonify,request
import numpy as np
import pandas as pd
import pickle
# Creating 

app=Flask(__name__)

# load model
model= pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict",methods=['POST'])
def predict():
    val=[i for i in request.form.values()]
    for i in range(len(val)):
        if i!=0 and i!=2:
            val[i]=float(val[i])
    features= [val]
    features= pd.DataFrame(features,columns=['Country', 'Year', 'Status', 'Adult Mortality', 'infant deaths',
       'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles', 'BMI',
       'under-five deaths', 'Polio', 'Total expenditure', 'Diphtheria',
       'HIV/AIDS', 'GDP', 'Population', 'thinness  1-19 years',
       'thinness 5-9 years', 'Income composition of resources', 'Schooling'])
    pred= model.predict(features)[0]
    return render_template("home.html",prediction_text="You will be alive {0} year on average".format(int(pred)))
if __name__=="__main__":
    app.run(debug=True)
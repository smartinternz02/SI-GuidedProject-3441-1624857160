import  numpy as np
import pandas as pd
from flask import Flask,request,render_template
import pickle
import os

app=Flask(__name__)
model=pickle.load(open('loanstatus.pkl','rb'))

@app.route('/')
def home():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [x for x in request.form.values()]
    features_value = [np.array(input_features[:-1])]
    features_name = ['Current Loan Amount', 'Term', 'Credit Score', 'annual income',
                     'years at work', 'Home Ownership', 'years of credit history',
                     'number of credit issues', 'Bankruptcies', 'Tax Liens',
                     'Credit Problems', 'Credit Age']
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    if output == 1:
        return render_template('output.html')
    else:
        return render_template('output2.html')
    
if __name__=='__main__':
    app.run(debug=True)
    
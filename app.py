from flask import Flask, jsonify, request, render_template

import numpy as np
import pandas as pd
import pickle
from model import *
#import nltk
#nltk.download('punkt', download_dir='/app/nltk_data/')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        
        user_input=[str(x) for x in request.form.values()]
        user_input=user_input[0]
        #print(user_input)
        output=predict_value(user_input)
        return render_template('index.html', prediction_text='Top 5 recommendations: {}'.format(output))
    else :
        return render_template('index.html')
    
if __name__ == '__main__':
    print('*** Recommendation system Started ***')
    app.run(debug=True)
    # app.run(host='127.0.0.1', port=5000)

    
    
    
    
    
    
    

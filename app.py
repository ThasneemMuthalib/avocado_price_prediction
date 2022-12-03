# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:15:20 2022

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 12:31:46 2022

@author: HP
"""
import numpy as np
import pickle
import math
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('model2.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    avo_type = request.form.get('type')
    avo_large_bags = math.log(float(request.form.get('large_bags'))+2)
    avo_date_new = request.form.get('date_new')
    avo_x4046 = math.log(float(request.form.get('x4046'))+2)
    avo_month_ex = request.form.get('month_ex')
    avo_x4225 = math.log(float(request.form.get('x4225'))+2)
    avo_region = request.form.get('region')
    avo_small_bags = math.log(float(request.form.get('small_bags'))+2)
    avo_total_volume = math.log(float(request.form.get('total_volume')))
    avo_year = request.form.get('year')
    avo_x4770 = math.log(float(request.form.get('x4770'))+2)
    avo_xlarge_bags = math.log(float(request.form.get('xlarge_bags'))+2)
    
    int_features = [avo_type,avo_large_bags,avo_date_new,avo_x4046,avo_month_ex,avo_x4225,
                    avo_region,avo_small_bags,avo_total_volume,avo_year,avo_x4770,avo_xlarge_bags]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    
    #prediction = model.predict([[request.form.get('type','large_bags','date_new','x4046',
    #'month_ex','x4225','region','small_bags','total_volume','year','x4770','xlarge_bags')]])
    # prediction = model.predict([[0,	10988593,136,11652577,8,10879205	,21,10800154,12555953,2017,8871842,5968580]])
    
    output = round(prediction[0],2)
    # print(output)
    return render_template('index.html', prediction_text = f'Average price Rs. $ {output}/-', 
                            large_bag=f'Large bags {avo_large_bags}',
                            avo_4770=f'avo_4770 {avo_x4770}')

if __name__ == '__main__':
    app.run(debug=True)
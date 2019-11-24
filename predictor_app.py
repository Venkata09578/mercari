import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from predictor_api import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    name =  request.form.get('itemname')
    shipping =  int(request.form.get('shipping'))
    item_description =  request.form.get('item_description')
    item_condition_id =  int(request.form.get('item_condition_id'))
    brandname =  request.form.get('brandname')
    maincategory =  request.form.get('category')
    n = type(shipping)

    input = {'item': [item_description],'category_name': [maincategory],'brand_name': [brandname],'name':[name],'shipping' : [shipping],'item_condition_id' : [item_condition_id]}
# #     #Converting to Df
    data1 = pd.DataFrame(input)
    output = price_output(data1)
# =============================================================================

    

    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
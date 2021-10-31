from flask import Flask, request, render_template
import numpy as np
import h2o
import pandas as pd
import math


h2o.init(port=54324)
model_path = 'GBM_model'
model = h2o.load_model(model_path)

col_dict = {'Age' : 'real',
            'PSA' : 'real',
            'PV': 'real',
            'DRE' : 'int', 
            'history' : 'int',
            'Inic_repeat' : 'int'            
            }


app = Flask(__name__)
@app.route('/')
def home():
    return render_template('prostate-html.html')

@app.route('/predict', methods=['POST'])
def predict():
    params = request.form.to_dict()
    
    age = float(params['Age'])
    psa = float(params['PSA'])
    pv = float(params['PV'])
    dre = int(params['DRE'])
    history = int(params['family_history'])
    biopsy = int(params['repeat_biopsy'])
        
    data = {}
    data['Age'] = age
    data['PV'] = pv
    data['DRE'] = dre
    data['Inic_repeat'] = biopsy
    data['history'] = history
    data['PSA'] = psa
    
    data = pd.DataFrame(data, index=[0])        
    data = h2o.H2OFrame(data, column_types = col_dict)   
    prediction = model.predict(data).as_data_frame()
    probability = float(prediction['p1'][0])*100.0
    
    return render_template('prostate-html.html',
                          Age='{}'.format(age),
                          PV='{}'.format(pv),
                          DRE='{}'.format(dre),
                          repeat_biopsy='{}'.format(biopsy),
                          family_history='{}'.format(history),
                          PSA='{}'.format(psa),
                          prediction_text ='Probability is {:.2f} %'.format(probability))


if __name__ == '__main__':
   #app.run(port = 8000, host = "localhost", debug = False)
   app.run()
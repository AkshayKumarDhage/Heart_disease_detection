from flask import Flask, render_template, url_for, request

import pandas as pd 
import numpy as np
import pickle 

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib 

app = Flask(__name__)

def get_values(value_list):
	
	#stdSclr = StandardScaler()
	#scaled = stdSclr.fit_transform(np.resize(value_list[:5], (1, 5) ))
	sex, cp, fbs, restecg, exang, slope, ca, thal = [0]*2, [0]*4, [0]*2, [0]*3, [0]*2, [0]*3, [0]*5, [0]*4
	sex[value_list[5]] = 1
	cp[value_list[6]] = 1
	fbs[value_list[7]] = 1
	restecg[value_list[8]] = 1
	exang[value_list[9]] = 1
	slope[value_list[10]] = 1
	ca[value_list[11]] = 1
	thal[value_list[12]] = 1
	return value_list[:5] + sex + cp + fbs + restecg + exang + slope + ca + thal

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	model = open('knn_model.pkl', 'rb')
	knn = joblib.load(model)

	if request.method == 'POST':
		values = request.form 
		values = dict(values)
		value_list = list(map(lambda x:float(x) if '.' in x else int(x), values.values()))

		final_values = get_values(value_list)
		resize_values = np.reshape(final_values, (1, 30))
		prediction = knn.predict(resize_values)
	return render_template('result.html', prediction=prediction, values = values)

if __name__ == '__main__':
	app.run(debug = 'True')
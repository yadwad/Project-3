# importing the packages

import json

import pandas as pd

import numpy as np

from flask import Flask, request, jsonify, render_template, flash#, redirect, url_for, session, logging

from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

from sklearn.externals import joblib

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



# value of __name__ should be  '__main__'

app = Flask(__name__)

# Loading model so that it works on production

model = joblib.load('model.pkl')


@app.route('/')

def index():

	# Index page

	return render_template('index.html')



@app.route('/about')

def about():

	# about me page

	return render_template('about.html')



class PredictorsForm(Form):

	"""

	This is a form class to retrieve the input from user through form



	Inherits: request.form class

	"""

	glucose = StringField(u'Glucose (For eg.: 148)', validators=[validators.input_required()])

	bloodpressure = StringField(u'Bloodpressure (For eg.: 148)', validators=[validators.input_required()])

	bmi = StringField(u'BMI (For eg.: 44)', validators=[validators.input_required()])

	skinthickness = StringField(u'Skinthickness (For eg.: 23)', validators=[validators.input_required()])

	insulin = StringField(u'Insulin (For eg.: 23)', validators=[validators.input_required()])

	diabetespedigreefunction = StringField(u'Diabetespedigreefunction (For eg.: 100)', validators=[validators.input_required()])

	age = StringField(u'Age (For eg.: 23)', validators=[validators.input_required()])



@app.route('/predict', methods=['GET', 'POST'])

def predict():

	form = PredictorsForm(request.form)

	

	# Checking if user submitted the form and the values are valid

	if request.method == 'POST' and form.validate():

		# Now save all values passed by user into variables

		glucose = form.glucose.data

		bloodpressure = form.bloodpressure.data

		bmi = form.bmi.data

		skinthicknesss = form.skinthickness.data

		insulin = form.insulin.data

		diabetespedigreefunction = form.diabetespedigreefunction.data

		age = form.age.data



		# Creating input for model for predictions

		predict_request = [int(glucose), int(bloodpressure), int(bmi), int(skinthicknesss), int(insulin), int(diabetespedigreefunction), int(age)]

		predict_request = np.array(predict_request).reshape(1, -1)



		# Class predictions from the model

		prediction = model.predict(predict_request)

		prediction = str(prediction[0])



		# Survival Probability from the model

		predict_prob = model.predict_proba(predict_request)

		predict_prob = str(predict_prob[0][1])



		# Passing the predictions to new view(template)

		return render_template('predictions.html', prediction=prediction, predict_prob=predict_prob)



	return render_template('predict.html', form=form)



@app.route('/train', methods=['GET'])

def train():

	# reading data

	df = pd.read_csv("data/data.csv")



	#defining predictors and label columns to be used

	predictors = ['Glucose','BloodPressure', 'BMI', 'Skinthickness', 'Insulin', 'DiabetesPedigreeFunction', 'Age']

	label = 'Outcome'



	#Splitting data into training and testing

	df_train, df_test, y_train, y_test = train_test_split(df[predictors], df[label], test_size=0.20, random_state=42)



	
	# Label encoding of object type predictors

	le = dict()

	for column in df_train.columns:

	    if df_train[column].dtype == np.object:

	        le[column] = LabelEncoder()

	        df_train[column] = le[column].fit_transform(df_train[column])



	# Applying same encoding from training data to testing data

	for column in df_test.columns:

	    if df_test[column].dtype == np.object:

	        df_test[column] = le[column].transform(df_test[column])



	# Initializing the model

	model = RandomForestClassifier(n_estimators=25, random_state=42)



	# Fitting the model with training data

	model.fit(X=df_train, y=y_train)



	# Saving the trained model on disk

	joblib.dump(model, 'model.pkl')



	# Return success message for user display on browser

	return 'Success'



if __name__ == '__main__':

	# Load the pre-trained model from the disk

	# model = joblib.load('model.pkl')

	# Running the app in debug mode allows to change the code and

	# see the changes without the need to restart the server

	app.run(debug=True)

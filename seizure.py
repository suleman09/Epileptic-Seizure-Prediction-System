from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

app = Flask(__name__)


# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
        return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file
    file = request.files['file']
    test = pd.read_csv(file)
    new_input2= [test.iloc[0, :178]]

    # Make predictions using the loaded model
    new_output = model.predict(new_input2)


   
    # Get the class label
    if new_output==[1]:
        result = 1
    else:
        result = 0

    # Render the results page
    return render_template('result.html', result=result)

    


if __name__ == '__main__':
    app.run(debug=True)

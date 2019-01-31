import os
from flask import Flask, redirect, render_template, request, jsonify
from flask_bootstrap import Bootstrap
import pickle
import numpy as np

app = Flask(__name__)
app.vars = {}
Bootstrap(app)

dir_path = '../data/'

# Load Pickled Models
with open(dir_path + 'tfidf_vectorizer.pkl', 'rb') as tv:
    tfidf_vectorizer = pickle.load(tv)

with open(dir_path + 'logistic_regression.pkl', 'rb') as m:
    classification_model = pickle.load(m)


@app.route('/')
def main():
    return redirect('/index')


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        type = int(request.form['type'])
        description = request.form['description']

        # Transform text data and combine
        vectorized_description = tfidf_vectorizer.transform([description])
        input_sample = np.append([age, gender, type], vectorized_description.todense())

        # Predict
        prob_of_survival = classification_model.predict_proba([input_sample])[0][1]
        percent_of_survival = round(prob_of_survival * 100, 2)

        # Save vars for re-rendering:
        app.vars['age'] = age
        if gender == 0:
            app.vars['gender'] = 'female'
        else:
            app.vars['gender'] = 'male'
        if type == 0:
            app.vars['type'] = 'elective'
        elif type == 1:
            app.vars['type'] = 'emergency'
        else:
            app.vars['type'] = 'urgent'
        app.vars['description'] = description.lower()

        return render_template('index.html', _anchor="predict",
                               age=app.vars['age'], gender=app.vars['gender'],
                               type=app.vars['type'], desc=app.vars['description'],
                               prediction_value=percent_of_survival)


if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))
    # basic_app.run(host='0.0.0.0', port=port)

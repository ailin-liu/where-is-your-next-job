import os
from flask import Flask, redirect, render_template, request, jsonify
from flask_bootstrap import Bootstrap
import pickle
#import dill
import numpy as np

app = Flask(__name__)
app.vars = {}
Bootstrap(app)

dir_path = 'models/'

# Load Pickled Models


with open(dir_path + 'eng_open_predict.pkl', 'rb') as pred:
    prediction_model = pickle.load(pred)
#prediction_model = dill.load(open(dir_path + 'eng_open_predict.dill', 'rb'))

with open(dir_path + 'LR_pipe_1vsall_test.pkl', 'rb') as pipe:
    eng_classifier = pickle.load(pipe)
#eng_classifier = dill.load(open(dir_path + 'LR_pipe_1vsall.dill', 'rb'))

@app.route('/')
def main():
    return redirect('/index')


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        major = request.form['major']
        skill1 = request.form['skill1']
        skill2 = request.form['skill2']
        skill3 = request.form['skill3']

        eng_titles = {0:'Aerospace',1:'Biomedical', 2:'Chemical', 3: 'Civil', 4:'Computer-Hardware', 
              5:'Electrical', 6:'Electronics', 7:'Environmental', 8:'Health-and-Safety',
              9:'Industrial', 10:'Material', 11:'Mechanical'}
        
        # combine text data 
        input_sample = [' '.join([major, skill1,skill2,skill3])]

        # Predict
        top_chance = sorted([(val,ind) for ind, val in enumerate(eng_classifier.predict_proba(input_sample)[0])],key = lambda x:x[0], reverse = True)
        percent_of_title1 = round(top_chance[0][0] * 100, 2)
        title1 = eng_titles[top_chance[0][1]]+' Engineer'
        
        percent_of_title2 = round(top_chance[1][0] * 100, 2)
        title2 = eng_titles[top_chance[1][1]]+' Engineer'
        
        top3_title1 = prediction_model[title1]
        top3_title2 = prediction_model[title2]

        # Save vars for re-rendering for prediction:
        app.vars['major'] = major
        app.vars['skill1'] = skill1.lower()
        app.vars['skill2'] = skill2.lower()
        app.vars['skill3'] = skill3.lower()
        if percent_of_title2 > 10:
            return render_template('index.html', _anchor="predict",
                               major=app.vars['major'], skill1=app.vars['skill1'],
                               skill2=app.vars['skill2'], skill3=app.vars['skill3'], desc=app.vars['desc'],
                               prediction_percent1 = percent_of_title1, title1 = title1, top3_title1 = top3_title1,
                               prediction_percent2 = percent_of_title2, title2 = title2, top3_title2 = top3_title2)
        else:
            return render_template('index.html', _anchor="predict",
                               major=app.vars['major'], skill1=app.vars['skill1'],
                               skill2=app.vars['skill2'], skill3=app.vars['skill3'], desc=app.vars['desc'],
                               prediction_percent1 = percent_of_title1, title1 = title1, top3_title1 = top3_title1,
                               prediction_percent2 = None, title2 = None, top3_title2 = None)
 

if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    # app.run()

from flask import Flask, render_template, url_for, redirect, request
from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField
from random import randint, seed
import time

from ml.classification import classify
from ml.relearning import relearn

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asdfghjk'



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')




class ClassificationForm(FlaskForm):
    r_value = IntegerField('R')
    g_value = IntegerField('G')
    b_value = IntegerField('B')

@app.route('/classification/', methods=['POST', 'GET'])
def classification():

    if request.method == 'POST':

        classification_form = ClassificationForm()
        if classification_form.validate_on_submit():

            r_value = classification_form.r_value.data
            g_value = classification_form.g_value.data
            b_value = classification_form.b_value.data
            result = classify(r_value, g_value, b_value)
            return render_template('classification/classification_done.html', r=r_value, g = g_value, b = b_value, result = result)
    

    return render_template('classification/classification.html', form_classification = ClassificationForm())




class RelearnForm(FlaskForm):

    choice = SelectField(choices=[('Black','Black'), ('Blue','Blue'), ('Brown', 'Brown'), ('Gray', 'Gray'), ('Green', 'Green'), ('Orange', 'Orange'), ('Red', 'Red'), ('Violet', 'Violet'), ('White', 'White'), ('Yellow', 'Yellow')])

    r_value = IntegerField('R')
    g_value = IntegerField('G')
    b_value = IntegerField('B')

@app.route('/relearning/', methods=['POST', 'GET'])
def relearning():

    if request.method == 'POST':         
        form_relearn = RelearnForm()

        if form_relearn.validate_on_submit():
            choice = form_relearn.choice.data
            r = form_relearn.r_value.data
            g = form_relearn.g_value.data
            b = form_relearn.b_value.data
            relearn(r, g, b, choice)
        return render_template('relearning/relearning_done.html', choice = choice, r= r, g = g, b = b)

    
    seed(time.time()*1000)
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)

    return render_template('relearning/relearning.html', r = r, g = g, b = b, form_relearn = RelearnForm())




app.run(host='0.0.0.0', port='8098', debug=True)
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

        color_hex = request.form.get('color')
        # TODO: Parse hex to r,g,b:
        color_hex = [1,2,3]
        r_value = color_hex[0]
        g_value = color_hex[1]
        b_value = color_hex[2]
        
        # classification_form = ClassificationForm()
        # if classification_form.validate_on_submit():
            # r_value = classification_form.r_value.data
            # g_value = classification_form.g_value.data
            # b_value = classification_form.b_value.data

        result = classify(r_value, g_value, b_value)
        # result = classify(color_hex)
        return render_template('classification/classification_done.html', r=r_value, g = g_value, b = b_value, result = result)
    

    return render_template('classification/classification.html', form_classification = ClassificationForm())




class RelearnForm(FlaskForm):

    choice = SelectField(choices=[
        ('Beige','Beige'),
        ('Blue','Blue'),
        ('Brown','Brown'),
        ('Green','Green'),
        ('Grey','Grey'),
        ('Orange','Orange'),
        ('Red','Red'),
        ('Violet','Violet'),
        ('Yellow','Yellow')])
  
    r_value = IntegerField('Red')
    g_value = IntegerField('Green')
    b_value = IntegerField('Blue')

@app.route('/relearning/', methods=['POST', 'GET'])
def relearning():
    form_relearn = RelearnForm()
    if request.method == 'GET': 
        seed(time.time()*1000)
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        form_relearn.r_value.data = r
        form_relearn.g_value.data = g
        form_relearn.b_value.data = b

        return render_template('relearning/relearning.html', r = r, g = g, b = b, form_relearn = form_relearn)

    if request.method == 'POST':         
        if form_relearn.validate_on_submit():
            choice = form_relearn.choice.data
            r = form_relearn.r_value.data
            g = form_relearn.g_value.data
            b = form_relearn.b_value.data
            relearn(r, g, b, choice)
        return render_template('relearning/relearning_done.html', choice = choice, r= r, g = g, b = b)

    





app.run(host='0.0.0.0', port='8098', debug=True)
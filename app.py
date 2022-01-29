'''
Script for setting up server
'''

from flask import Flask, render_template, url_for, redirect, request
from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField
from random import randint, seed
import time

from ml.classification import classify
from ml.relearning import relearn

from PIL import ImageColor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asdfghjk'


@app.route('/', methods=['GET'])
def index():
    '''
    Main index endpoint
    '''
    return render_template('index.html')


class ClassificationForm(FlaskForm):
    ''' class for rgb flaskform'''
    r_value = IntegerField('R')
    g_value = IntegerField('G')
    b_value = IntegerField('B')

@app.route('/classification/', methods=['POST', 'GET'])
def classification():
    '''Classification endpoint'''

    if request.method == 'POST':

        color = request.form['color']
        rgb = ImageColor.getcolor(color, "RGB")
        r_color = rgb[0]
        g_color = rgb[1]
        b_color = rgb[2]

        result = classify(r_color, g_color, b_color)
        return render_template('classification/classification_done.html',
        r=r_color, g=g_color, b=b_color, result=result)

    return render_template('classification/classification.html',
    form_classification=ClassificationForm())


class RelearnForm(FlaskForm):
    '''Class for categorizing flaskform'''

    choice = SelectField(choices=[
        ('Beige', 'Beige'),
        ('Blue', 'Blue'),
        ('Brown', 'Brown'),
        ('Green', 'Green'),
        ('Grey', 'Grey'),
        ('Orange', 'Orange'),
        ('Red', 'Red'),
        ('Violet', 'Violet'),
        ('Yellow', 'Yellow')])

    r_value = IntegerField('Red')
    g_value = IntegerField('Green')
    b_value = IntegerField('Blue')


@app.route('/relearning/', methods=['POST', 'GET'])
def relearning():
    '''Relearning endpoint'''
    form_relearn = RelearnForm()
    if request.method == 'GET':
        seed(time.time()*1000)
        r_color = randint(0, 255)
        g_color = randint(0, 255)
        b_color = randint(0, 255)
        form_relearn.r_value.data = r_color
        form_relearn.g_value.data = g_color
        form_relearn.b_value.data = b_color

        return render_template('relearning/relearning.html',
        r=r_color, g=g_color, b=b_color, form_relearn=form_relearn)

    if request.method == 'POST':
        if form_relearn.validate_on_submit():
            choice = form_relearn.choice.data
            r_color = form_relearn.r_value.data
            g_color = form_relearn.g_value.data
            b_color = form_relearn.b_value.data
            relearn(r_color, g_color, b_color, choice)
        return render_template('relearning/relearning_done.html',
        choice=choice, r=r_color, g=g_color, b=b_color)


app.run(host='0.0.0.0', port='8098', debug=True)

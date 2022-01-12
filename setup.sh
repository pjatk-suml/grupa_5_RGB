#!/bin/bash
python -m venv env
source env/Scripts/activate
pip install flask
pip install flask_wtf
pip install wtforms
pip install sklearn
pip install pandas
pip install numpy
python app.py
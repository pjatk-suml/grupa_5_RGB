#!/bin/bash
python -m venv env
#source env/Scripts/activate
source env/bin/activate
pip install flask
pip install flask_wtf
pip install wtforms
pip install sklearn
pip install pandas
pip install numpy
pip install Pillow
pip install tensorflow==2.2.0
pip install keras==2.3.1
python app.py
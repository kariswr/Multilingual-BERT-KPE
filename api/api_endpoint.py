from flask import Flask
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api, reqparse
from os.path import dirname
import ast
import sys
import pandas as pd

from answer_extractor import ExtractAnswer

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(app)

api.add_resource(ExtractAnswer, '/extract-answer')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)
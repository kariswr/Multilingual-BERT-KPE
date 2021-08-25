from flask import Flask
from flask_restful import Resource, Api, reqparse
from os.path import dirname
import ast
import sys
import pandas as pd

# sys.path.insert(1, dirname(dirname(sys.path[0])))
from answer_extractor import ExtractAnswer

app = Flask(__name__)
api = Api(app)

api.add_resource(ExtractAnswer, '/extract-answer')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug = True)
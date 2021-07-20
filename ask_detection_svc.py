from flask import Flask, request
  
from ask_classifier import AskClassifier


app = Flask(__name__)


@app.route('/')
def index():
    return '*** Ask detection service *** \n'


@app.route('/classify', methods=['GET'])
def classify():
    print(f"******************************************")
    print(request.__dict__)
    print(f"******************************************")
    #print(request.environ['werkzeug'].request.__dict__)
    print(f"******************************************")
    text = request.args.get("text")
    return str(AskClassifier.is_an_ask(text))

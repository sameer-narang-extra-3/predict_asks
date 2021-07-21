import logging

import google.cloud.logging

from flask import Flask, request
  
from ask_classifier import AskClassifier


app = Flask(__name__)

gunicorn_error_logger = logging.getLogger('gunicorn.error')
app.logger.handlers.extend(gunicorn_error_logger.handlers)
app.logger.setLevel(logging.INFO)

client = google.cloud.logging.Client()
client.setup_logging()


@app.route('/')
def index():
    return '*** Ask detection service *** \n'


@app.route('/classify', methods=['GET'])
def classify():
    text = request.args.get("text")
    rcode = 0
    is_an_ask = False
    try:
        is_an_ask = AskClassifier.is_an_ask(text)
    except Exception as exc:
        app.logger.error(f"Exception encountered while classifying '{text}' --- {exc}")
        print(f"Exception encountered while classifying '{text}' --- {exc}")
        rcode = 1
    return {"rcode": rcode, "is_an_ask": is_an_ask}


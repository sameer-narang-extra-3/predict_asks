import json
import os
import re
import torch

import google.cloud.logging

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


client = google.cloud.logging.Client()
client.setup_logging()
logger = client.logger(__name__)

BASE_DIR = "."


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model_dir = os.path.join(BASE_DIR, "model")
logger.log_text(f"Loading model from: {model_dir}")
model = DistilBertForSequenceClassification.from_pretrained(model_dir)
logger.log_text(f"Model loaded")
softmax = torch.nn.Softmax(dim=1)

class AskClassifier(object):
    @staticmethod
    def clean_message(msg: str) -> str:
        """
        should strip out user identifiers and replace with <mentioned_user>
        should also strip out channel identifiers and replace with <mentioned_channel>
        """
        msg = re.sub(r"<@[A-Za-z0-9|]+[\w\.-]*?>", "<SlackUserId>", msg)
        msg = re.sub(r"[<]?http\S+", "<SomeHttpOrHttpsUrl>", msg)
        return msg

    @staticmethod
    def is_an_ask(msg:str) -> bool:
        logger.log_text(f"classifying msg = '{msg}' ---- {type(msg)}")
        print(f"classifying msg = '{msg}' ---- {type(msg)}")
        inputs = tokenizer(msg, padding=True, truncation=True, return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)

        msg = AskClassifier.clean_message(msg)

        outputs = model(**inputs, labels=labels)
        normalized_outputs = softmax(outputs.logits)
        logger.log_text(f"classifier output: {normalized_outputs} ----- {msg}")
        print(f"classifier output: {normalized_outputs} ----- {msg}")
        return normalized_outputs.tolist()[0][1] > 0.55


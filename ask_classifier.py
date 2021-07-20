import json
import os
import re
import torch

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


BASE_DIR = "."


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model_dir = os.path.join(BASE_DIR, "model")
#logger.info(f"Loading model from: {model_dir}")
model = DistilBertForSequenceClassification.from_pretrained(model_dir)
#logger.info(f"Model loaded")
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
        print(f"msg = '{msg}'----{type(msg)}")
        rcode = 0
        is_an_ask = False
        try:
            inputs = tokenizer(msg, padding=True, truncation=True, return_tensors="pt")
            labels = torch.tensor([1]).unsqueeze(0)

            msg = AskClassifier.clean_message(msg)

            outputs = model(**inputs, labels=labels)
            normalized_outputs = softmax(outputs.logits)
            #logger.info(f"classifier output: {normalized_outputs} ----- for {msg}")
            print(f"classifier output: {normalized_outputs} ----- {msg}")
            is_an_ask = normalized_outputs.tolist()[0][1] > 0.55
        except Exception as exc:
            print(f"Exception encountered while classifying '{msg}' --- {exc}")
            rcode = 1
        return {"rcode": rcode, "is_an_ask": is_an_ask}


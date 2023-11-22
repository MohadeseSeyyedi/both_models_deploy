from flask import Flask, request, render_template
import torch
from transformers import BertConfig, BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification
from parsbert_predict import parsbert_predict
from parsbert_architecture import SentimentModel

from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaForSequenceClassification
from absa_predict import absa_predict

from paths import *
from detect_lang import detect_lang

#Create an app object using the Flask class. 
app = Flask(__name__)

# Load Aspect-Based Sentiment Analysis model
absa_tokenizer = AutoTokenizer.from_pretrained(absa_tokenizer_path, local_files_only=True)
absa_model = AutoModelForSequenceClassification.from_pretrained(absa_model_path, local_files_only=True)

# Load basic sentiment analysis model
sentiment_tokenizer = XLMRobertaTokenizerFast.from_pretrained(sentiment_tokenizer_path, local_files_only=True)
sent_model = XLMRobertaForSequenceClassification.from_pretrained(sentiment_model_path, local_files_only=True)

# Load Parsbert model
parsbert_config = BertConfig.from_pretrained(parsbert_config_path, local_files_only=True)
parsbert_tokenizer = BertTokenizer.from_pretrained(parsbert_tokenizer_path, local_files_only=True)
parsbert_model = SentimentModel(parsbert_config)
parsbert_model.load_state_dict(torch.load(parsbert_state_dict_path, map_location=torch.device('cpu')))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def show_prediction():
    from numpy import round
    comment = request.form['comment']
    aspect = request.form['aspect']
    comment_lang = detect_lang(comment)
    
    if comment_lang == 'en':
        if aspect:
            prediction_dict = absa_predict(comment, aspect, absa_tokenizer, absa_model)
            positive = round(prediction_dict["positive"], 2)
            negative = round(prediction_dict["negative"], 2)
            neutral = round(prediction_dict["neutral"], 2)
            
            prediction_text = f'positive:{round(100*positive, 2)}\t\t\tnegative:{round(100*negative, 2)}\t\t\tneutral:{round(100*neutral, 2)}'

        else:
            prediction_dict = absa_predict(comment, None, sentiment_tokenizer, sent_model)
            for key, value in prediction_dict.items():
                if value:
                    prediction_text = f'{round(value*100, 2)}% {key}'

    else:
        prediction_dict = parsbert_predict(parsbert_model, [comment], parsbert_tokenizer)
        positive = round(prediction_dict["positive"], 2)
        negative = round(prediction_dict["negative"], 2)
        prediction_text = f'positive:{round(100*positive, 2)}\t\t\tnegative:{round(100*negative, 2)}'
    
    return render_template('index.html', input_comment=comment, aspect_text = prediction_dict['aspect'], prediction_text=prediction_text)


if __name__=='__main__':
    app.run()












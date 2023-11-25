from huggingface_to_local import *
from paths import *

from transformers import BertModel, BertConfig, BertTokenizer
from transformers import BertConfig, BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaForSequenceClassification


# Download Parsbert
load_config(parbert_hugguingface, parsbert_config_path, BertConfig)
load_model(parbert_hugguingface, parsbert_model_path, BertModel)
load_tokenizer(parbert_hugguingface, parsbert_tokenizer_path, BertTokenizer)

# Download Sentiment
load_model(sentiment_huggingface, sentiment_model_path, XLMRobertaForSequenceClassification)
load_tokenizer(sentiment_huggingface, sentiment_tokenizer_path, XLMRobertaTokenizerFast)

# Download ABSA
load_model(absa_huggingface, absa_model_path, AutoModelForSequenceClassification)
load_tokenizer(absa_huggingface, absa_tokenizer_path, AutoTokenizer)
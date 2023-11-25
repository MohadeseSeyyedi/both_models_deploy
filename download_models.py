from huggingface_to_local import *
from paths import *

# Download Parsbert
load_config(parbert_hugguingface, parsbert_config_path)
load_model(parbert_hugguingface, parsbert_model_path)
load_tokenizer(parbert_hugguingface, parsbert_tokenizer_path)

# Download Sentiment
load_config(sentiment_huggingface, sentiment_config_path)
load_model(sentiment_huggingface, sentiment_model_path)
load_tokenizer(sentiment_huggingface, sentiment_tokenizer_path)

# Download ABSA
load_config(absa_huggingface, absa_config_path)
load_model(absa_huggingface, absa_model_path)
load_tokenizer(absa_huggingface, absa_tokenizer_path)
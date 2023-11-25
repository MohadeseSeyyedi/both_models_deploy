

def load_model(huggingface_path, local_path, model_class):
    model_class.from_pretrained(huggingface_path).save_pretrained(local_path)

def load_config(huggingface_path, local_path, model_class):
    model_class.from_pretrained(huggingface_path).save_pretrained(local_path)

def load_tokenizer(huggingface_path, local_path, model_class):
    model_class.from_pretrained(huggingface_path).save_pretrained(local_path)


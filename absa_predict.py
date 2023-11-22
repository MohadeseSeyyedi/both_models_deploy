from transformers import pipeline
import torch.nn.functional as F

def absa_predict(comment, aspect, tokenizer, model):
    output_dict = {'aspect': aspect, 'negative':None, 'neutral':None, 'positive':None}
    if aspect:
        inputs = tokenizer(f"[CLS] {comment} [SEP] {aspect} [SEP]", return_tensors="pt")
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        probs = probs.detach().numpy()[0]
        output_dict = {'aspect': aspect, 'negative':probs[0], 'neutral':probs[1], 'positive':probs[2]}
    else:
        sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        sentiment = sentiment_pipe([comment])[0]
        output_dict[sentiment['label']] = sentiment['score']
    
    return output_dict
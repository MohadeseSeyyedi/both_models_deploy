import torch
from tqdm import tqdm
from parsbert_dataset import create_data_loader
import torch.nn.functional as F


def parsbert_predict(model, comments, tokenizer, max_len=128, batch_size=32):
    data_loader = create_data_loader(comments, None, tokenizer, max_len, batch_size, None)
    predictions = []
    prediction_probs = []

    model.eval()
    with torch.no_grad():
        for dl in tqdm(data_loader, position=0):
            input_ids = dl['input_ids']
            attention_mask = dl['attention_mask']
            token_type_ids = dl['token_type_ids']

            # compute predicted outputs by passing inputs to the model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
            
            # convert output probabilities to predicted class
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds)
            prediction_probs.extend(F.softmax(outputs, dim=1))

    predictions = torch.stack(predictions).cpu().detach().numpy()
    prediction_probs = torch.stack(prediction_probs).cpu().detach().numpy()

    
    output_dict = {'aspect': None, 'negative':None, 'neutral':None, 'positive':None}
    output_dict['negative'] = prediction_probs[0][0]
    output_dict['positive'] = prediction_probs[0][1]

    return output_dict
    
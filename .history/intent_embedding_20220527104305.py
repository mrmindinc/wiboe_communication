import torch
import numpy as np
from cosine_similarity import cosine_similarity_check

device = torch.device("cpu")
model = torch.load("SAVE_MODEL_DIR/model", map_location=device)

def text_similarity(data):
    intent_list = data["intent_list"]
    intent_id = data["intent_id"]
    text = data["text"]
    
    intent_embedding = model.encode(intent_list)
    
    score = []
    for embedding in intent_embedding:
        statement_input_embedding = model.encode(text)
        similarity = cosine_similarity_check(statement_input_embedding, embedding)
        score.append(similarity)
        
    percent= score[np.argmax(score)].tolist()
    percent = round(percent, 4)

    if percent >= 45 :
        response_node = {
            'node_id': intent_id[np.argmax(score)],
            'node_text': intent_list[np.argmax(score)],
            'confidence': percent
        }

    else :
        response_node = {
                'node_id': 700,
                'node_text': '무응답',
                'confidence': percent
        }

    return response_node, percent
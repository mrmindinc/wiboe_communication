import torch
import numpy as np
from adapter import Adapter
from cosine_similarity import cosine_similarity_check, levenshtein_distance
import numpy as np


class ExtractIntent(Adapter):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def text_similarity(self, data):
        intent_list = data["intent_list"]
        intent_id = data["intent_id"]
        text = data["text"]
        
        if len(intent_list) > 5:
            confidence_list = []
            for intent in intent_list:
                confidence = levenshtein_distance(text, intent)
                confidence_list.append(confidence)
            
            
            sorted_indexs = np.argsort(confidence_list)[::-1]
            
            new_intent_list, new_intent_id = [], []
            for sorted_index in sorted_indexs:
                new_intent_list.append(intent_list[sorted_index])
                new_intent_id.append(intent_id[sorted_index])
                if len(new_intent_list) >= 5:
                    break
                
            intent_list = new_intent_list
            intent_id = new_intent_id

        intent_embedding = self.model.encode(intent_list)

        score = []
        for embedding in intent_embedding:
            statement_input_embedding = self.model.encode(text)
            similarity = cosine_similarity_check(statement_input_embedding, embedding)
            score.append(similarity)
            
        print(score)

        percent = score[np.argmax(score)].tolist()
        percent = round(percent, 4)

        if percent >= 0.45 :
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

        print(response_node)
        print(data["intent_list"])
        print(score, np.argmax(score))

        return response_node


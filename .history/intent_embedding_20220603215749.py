import torch
import numpy as np
from adapter import Adapter
from cosine_similarity import cosine_similarity_check


# class ExtractIntent(Adapter):
#     def __init__(self, model, **kwargs):
#         super().__init__(model, **kwargs)

#     def text_similarity(self, data):
#         intent_list = data["intent_list"]
#         intent_id = data["intent_id"]
#         text = data["text"]

#         import time
#         start = time.time()

#         intent_embedding = self.model.encode(intent_list)

#         end = time.time()
#         pre = end - start 

#         score = []
#         for embedding in intent_embedding:
#             statement_input_embedding = self.model.encode(text)
#             similarity = cosine_similarity_check(statement_input_embedding, embedding)
#             score.append(similarity)
            
#         print(score)

#         percent = score[np.argmax(score)].tolist()
#         percent = round(percent, 4)

#         if percent >= 0.45 :
#             response_node = {
#                 'node_id': intent_id[np.argmax(score)],
#                 'node_text': intent_list[np.argmax(score)],
#                 'confidence': percent
#             }

#         else :
#             response_node = {
#                     'node_id': 700,
#                     'node_text': '무응답',
#                     'confidence': percent
#             }

#         print(response_node)
#         print(data["intent_list"])
#         print(score, np.argmax(score))

#         return response_node

class ExtractIntent(Adapter):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def text_similarity(self, data):
        intent_list = data["intent_list"]
        intent_id = data["intent_id"]
        text = data["text"]

        import time
        start = time.time()
        
        intent_embedding = self.model.encode(intent_list)

        end = time.time()
        pre = end - start 

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
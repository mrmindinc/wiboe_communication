import numpy as np
from cosine_similarity import cosine_similarity_check
from model_adapters import Adapter
import numpy as np


class ExtractIntent(Adapter):
    def __init__(self, model):
        super().__init__(model)

    def text_similarity(self, data):
        intent_list = data["intent_list"]
        intent_id = data["intent_id"]
        text = data["text"]

        intent_embedding = self.model.encode(intent_list)

        score = []
        for embedding in intent_embedding:
            statement_input_embedding = self.model.encode(text)
            similarity = cosine_similarity_check(statement_input_embedding, embedding)
            score.append(similarity)

        percent = score[np.argmax(score)].tolist()
        percent = round(percent, 4)

        if percent >= 0.45 :
            response_node = {
                'node_id': int(intent_id[np.argmax(score)]),
                'node_text': intent_list[np.argmax(score)],
                'confidence': percent
            }

        else :
            response_node = {
                    'node_id': int(700),
                    'node_text': '무응답',
                    'confidence': 0.0
            }

        print("response_node: ", response_node)
        print("intent_list: ", data["intent_list"])
        print("score: ", score)

        return response_node


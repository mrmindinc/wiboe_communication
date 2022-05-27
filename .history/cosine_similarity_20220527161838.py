from sklearn.metrics.pairwise import cosine_similarity

class Comparator:
    
    def __call__(self, statement_a, statement_b):
        return self.compare(statement_a, statement_b)

    def compare(self, statement_a, statement_b):
        return 0

class CosineSimilarity(Comparator):
    
    def compare(self, text, other_text):
        import time
        start = time.time()
        
        similarity = cosine_similarity([text], [other_text]).squeeze()

        end = time.time()
        pre = end - start
        
        return similarity
    
cosine_similarity_check = CosineSimilarity()
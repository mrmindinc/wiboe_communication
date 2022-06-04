from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

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
    
class LevenshteinDistance(Comparator):
    """
    Compare two statements based on the Levenshtein distance
    of each statement's text.

    For example, there is a 65% similarity between the statements
    "where is the post office?" and "looking for the post office"
    based on the Levenshtein distance algorithm.
    """

    def compare(self, text, other_text):
        """
        Compare the two input statements.

        :return: The percent of similarity between the text of the statements.
        :rtype: float
        """

        # Return 0 if either statement has a falsy text value
        if not text or not other_text:
            return 0

        # Get the lowercase version of both strings
        text = str(text.lower())
        other_text = str(other_text.lower())

        similarity = SequenceMatcher(
            None,
            text,
            other_text
        )

        # Calculate a decimal percent of the similarity
        percent = round(similarity.ratio(), 2)

        return percent
    
cosine_similarity_check = CosineSimilarity()
levenshtein_distance = LevenshteinDistance()
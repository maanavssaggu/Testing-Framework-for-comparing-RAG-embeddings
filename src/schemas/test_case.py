class TestCase:
    def __init__(self, qa_pair: dict):
        self.question=qa_pair['QA'].question
        self.answer=qa_pair['QA'].answer
        self.doc_id=qa_pair['doc_id']
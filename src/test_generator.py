
import random
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.docstore.document import Document
from typing import List, Dict
from src.schemas.question import Question
from src.schemas.test_case import TestCase
from src.vectorstore import VectorStoreManager, SqlDb
from src.model import Model
from src.pipeline import Pipeline
        

class TestQuestionGenerator:
    def __init__(self):
        self.llm = Model()
        self.vector_store_manager = VectorStoreManager()
        self.pipeline = Pipeline(model=self.llm, vector_store_manager=self.vector_store_manager)
        self.sql = SqlDb()
        pass

    def pick_random_document(self)->Dict[str, str]: 
        """
        returns a random dictionary with the Document contents and doc id 
        """
        ret = {}
        ids = self.vector_store_manager.vector_store.get()['ids']
        if not ids:
            self.pipeline.process_data()
        random_id = random.choice(ids)
        ret['doc_id'] = random_id
        ret['document'] = self.vector_store_manager.vector_store.get(random_id)['documents'] 
        return ret
    
    def generate_test_case(self, document_content: str)->TestCase:
        print(document_content)
        try: 
            if not self.sql.doc_id_has_question(document_content['doc_id']):
                print(f"generating a new QA pair for {document_content['doc_id']}")
                qa = self.llm.generate_qa_pair(doc_id=document_content['doc_id'], document_content=document_content["document"])
                self.sql.insert_question(qa)
            else:
                print(f"getting question for doc {document_content['doc_id']} from db")
                qa = self.sql.get_question_by_doc_id(document_content['doc_id'])

            return qa
    
        except Exception as e:
            print(f"error occurred generating the question {e}")

    def run_test_case(self, pipeline_to_test: Pipeline, test_case: TestCase)->bool:
        question = test_case.question
        # ask pipeline the question
        response, sources = pipeline_to_test.generate(input_query=question)
        sources = set(sources)

        if test_case.doc_id in sources:
            print(f'passed test: {question}')
            return True 
        else:
            print(f'failed test: {question}')
            return False    

        
        

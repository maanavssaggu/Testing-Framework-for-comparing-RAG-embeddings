import random
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.docstore.document import Document
from typing import List, Dict
from src.schemas.test_case import TestCase
from src.schemas.question import Question
from src.vectorstore import VectorStoreManager, SqlDb
from src.model import Model
from src.pipeline import Pipeline

class TestQuestionGenerator:
    def __init__(self):
        """
        Initialize the TestQuestionGenerator with components for generating and testing questions.
        """
        self.llm = Model()
        self.vector_store_manager = VectorStoreManager()
        self.pipeline = Pipeline(model=self.llm, vector_store_manager=self.vector_store_manager)
        self.sql = SqlDb()

    def pick_random_document(self) -> Dict[str, str]: 
        try:
            ret = {}
            # Get document IDs from the vector store
            ids = self.vector_store_manager.vector_store.get()['ids']
            if not ids:
                self.pipeline.process_data()
            random_id = random.choice(ids)

            # Store the selected document's ID and content
            ret['doc_id'] = random_id
            ret['document'] = self.vector_store_manager.vector_store.get(random_id)['documents'] 
            return ret
        except Exception as e:
            print(f"Error in TestQuestionGenerator.pick_random_document: {e}")
            return {}

    def generate_test_case(self, document_content: str) -> TestCase:
        """
        Generates or retrieves a test case for a given document.
        """
        try: 
            if not self.sql.doc_id_has_question(document_content['doc_id']):
                # Generate a new QA pair using the LLM
                qa = self.llm.generate_qa_pair(doc_id=document_content['doc_id'], document_content=document_content["document"])
                self.sql.insert_question(qa)
            else:
                # Retrieve existing QA pair from the database
                qa = self.sql.get_question_by_doc_id(document_content['doc_id'])
            print(f"Generated a QA pair for {document_content['doc_id']}")
            return qa
        except Exception as e:
            print(f"Error in TestQuestionGenerator.generate_test_case: {e}")
            return None

    def run_test_case(self, pipeline_to_test: Pipeline, test_case: TestCase) -> bool:
        """
        Executes a test case by querying the pipeline and verifying the result.
        """
        try:
            question = test_case.question
            # Ask the pipeline the test question
            response, sources = pipeline_to_test.generate(input_query=question)
            sources = set(sources)

            # Check if the correct document ID is among the sources
            if test_case.doc_id in sources:
                print(f'Passed test: {question}')
                return True 
            else:
                print(f'Failed test: {question}')
                return False    
        except Exception as e:
            print(f"Error in TestQuestionGenerator.run_test_case: {e}")
            return False

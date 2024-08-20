"""
This file, creates the db instance and allows you to populate the db with the documents 

need to take in the embedding model as a parameter for the class, that way you can have
two seperate instances of a database 
"""
import os 
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from pydantic import BaseModel
from typing import Optional, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from src.schemas.question import Question
from src.schemas.test_case import TestCase
# from test_generator import Question

import sqlite3

PERSITENT_DIR_PATH = "db/chroma_langchain_db"

class VectorStoreManager():
    def __init__(self, embedding_function:Optional[Embeddings]=OpenAIEmbeddings(model="text-embedding-3-large")) -> None:
        print("initilising vector store")
        self.embedding_function=embedding_function
        self.collection_name = "llm-embedding-test-suite-1"
        self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=80,
                    length_function=len,
                    is_separator_regex=False,
                )
        self.sql_document_tracker = SqlDb()

        try:
            self.client = chromadb.PersistentClient()
            self.collection = self.client.get_or_create_collection("main-collection")
            self.embedding_function = self.embedding_function
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=PERSITENT_DIR_PATH,  # Where to save data locally, remove if not neccesary
            )
            self.embedding = self.embedding_function.model
            print("succesfully initilised vector store")
        except Exception as e:
            print(f"an Exception {e} occured when initilising the vector store")
        
            
    def ingest_data(self):
        """
        checks data folder for any documents and uploads them to the vector store
        """ 
        print("Checking to see if any new documents have been added")
        
        # check for new files 
        knowledge_files = os.listdir('data/')
        new_knowledge = set()

        knowledge_file_tracker_db = SqlDb()

        for file in knowledge_files:
            if not knowledge_file_tracker_db.document_with_embedding_exists(doc_id=file, embedding=self.embedding):
                new_knowledge.add(file)
                print(f"new file found {file} with embedding: {self.embedding}")
                self.sql_document_tracker.insert_document_and_embedding(name=file, embedding=self.embedding)
            
        # start to upload the new files 
        if new_knowledge:
            for knowledge in new_knowledge:
                print(f"data/{knowledge}")
                loader = PyPDFLoader(f"data/{knowledge}")
                documents = loader.load_and_split()
                split_documents = self.text_splitter.split_documents(documents)
                self.add_to_chroma(split_documents)
        else:
            print('no new data to be added')
        

    def update_db(self,):
        pass

    def add_to_chroma(self, chunks):
        vs = self.vector_store

        # Calculate Page IDs.
        chunks_with_ids = self.calculate_chunk_ids(chunks)

        # Add or Update the documents.
        existing_items = vs.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in vs: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            print(new_chunks)
            vs.add_documents(documents=new_chunks, ids=new_chunk_ids)
            print("successfully added new documents")
        else:
            print("No new documents to add")

    def calculate_chunk_ids(self, chunks):

        # This will create IDs like "doc: POL011BA.pdf page: 6:2"
        # Page Source : Page Number : Chunk Index

        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            chunk.metadata['embedding']=self.embedding
            source = chunk.metadata.get("source")[4:]
            page = chunk.metadata.get("page")
            current_page_id = f"doc: {source} page:{page}"

            # increment chunk index if on same page
            current_chunk_index = current_chunk_index + 1 if last_page_id == current_page_id else 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id

            last_page_id = current_page_id

        return chunks
  
class SqlDb:
    def __init__(self) -> None:
        try:
            # No need to store the connection in self.conn
            with sqlite3.connect('db/knowledge_files_tracker2.db') as conn:
                cursor = conn.cursor()

                # Create a table to store documents and embeddings (if it doesn't already exist)
                cursor.execute('''CREATE TABLE IF NOT EXISTS documents (
                                    id INTEGER PRIMARY KEY,
                                    doc_title TEXT,
                                    embedding_name TEXT,
                                    UNIQUE(doc_title, embedding_name)
                                )''')
                # Create a table to store document IDs and associated questions and answers
                cursor.execute('''CREATE TABLE IF NOT EXISTS qa_pairs (
                                    id INTEGER PRIMARY KEY,
                                    doc_id TEXT,
                                    question TEXT NOT NULL,
                                    answer TEXT NOT NULL,
                                    FOREIGN KEY (doc_id) REFERENCES documents(doc_title)
                                )''')
                conn.commit()
            print("Successfully instantiated SQL db")
        except Exception as e:
            print(f"Error occurred when instantiating the SQL db: {e}")

    def insert_document_and_embedding(self, name: str, embedding: str) -> None:
        try:
            with sqlite3.connect('db/knowledge_files_tracker2.db') as conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO documents (doc_title, embedding_name) VALUES (?, ?)', (name, embedding))
                conn.commit()
        except sqlite3.IntegrityError:
            pass  # Document with this title and embedding already exists
        except Exception as e:
            print(f"An error occurred: {e}")

    def insert_question(self, test_case: TestCase) -> None:
        try:
            with sqlite3.connect('db/knowledge_files_tracker2.db') as conn:
                cursor = conn.cursor()
                doc_id = test_case.doc_id
                question = test_case.question
                answer = test_case.answer
                cursor.execute(
                    'INSERT INTO qa_pairs (doc_id, question, answer) VALUES (?, ?, ?)', 
                    (doc_id, question, answer)
                )
                conn.commit()
            print(f"Successfully inserted question-answer pair for document ID {doc_id}.")
        except Exception as e:
            print(f"Error occurred while inserting question-answer pairs into SQL db: {e}")

    def document_with_embedding_exists(self, doc_id: str, embedding: str) -> bool:
        try:
            with sqlite3.connect('db/knowledge_files_tracker2.db') as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1 FROM documents WHERE doc_title = ? AND embedding_name = ?', (doc_id, embedding))
                conn.commit()
            return bool(cursor.fetchone())
        except Exception as e:
            print(f"Error occurred checking if document: {doc_id} with embedding {embedding} exists - error:   {e}")


    def get_question_by_doc_id(self, doc_id: str) -> TestCase:
        try:
            with sqlite3.connect('db/knowledge_files_tracker2.db') as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT question, answer FROM qa_pairs WHERE doc_id = ?', (doc_id,))
                row = cursor.fetchone()
                if row:
                    test_case = {
                        "QA": Question(question=row[0], answer=row[1]),
                        "doc_id": doc_id
                    }
                    return TestCase(test_case)
                else:
                    raise ValueError(f"No question found for doc_id: {doc_id}")
        except Exception as e:
            print(f"Error occurred while retrieving question-answer pairs: {e}")
            return None

    def doc_id_has_question(self, doc_id: str) -> bool:
        try:
            with sqlite3.connect('db/knowledge_files_tracker2.db') as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1 FROM qa_pairs WHERE doc_id = ? LIMIT 1', (doc_id,))
                result = cursor.fetchone()
                return result is not None
        except Exception as e:
            print(f"Error occurred while checking if doc_id {doc_id} has questions: {e}")
            return False

    def get_all_entries(self, embedding: str) -> list:
        try:
            with sqlite3.connect('db/knowledge_files_tracker2.db') as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT doc_title FROM documents WHERE embedding_name = ?', (embedding,))
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error occurred while getting all entries: {e}")
            return []

    def delete_entry(self, doc_id: str, embedding: str) -> None:
        try:
            with sqlite3.connect('db/knowledge_files_tracker2.db') as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM documents WHERE doc_title = ? AND embedding_name = ?', (doc_id, embedding))
                conn.commit()
        except Exception as e:
            print(f"Error occurred while deleting document: {e}")





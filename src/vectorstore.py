import os 
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from pydantic import BaseModel
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from src.schemas.question import Question
from src.schemas.test_case import TestCase
import sqlite3

PERSITENT_DIR_PATH = "db/chroma_langchain_db"

class VectorStoreManager:
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
                persist_directory=PERSITENT_DIR_PATH, 
            )
            self.embedding = self.embedding_function.model
            print("succesfully initilised vector store")
        except Exception as e:
            print(f"Error in VectorStoreManager.__init__: {e}")
        
    def ingest_data(self):
        """
        Ingest new documents from 'data/' folder into the vector store.
        """ 
        print("Checking to see if any new documents have been added")
        try:
            # get files
            knowledge_files = os.listdir('data/')
            new_knowledge = set()
            knowledge_file_tracker_db = SqlDb()

            # check if any new files
            for file in knowledge_files:
                if not knowledge_file_tracker_db.document_with_embedding_exists(doc_id=file, embedding=self.embedding):
                    new_knowledge.add(file)
                    print(f"new file found {file} with embedding: {self.embedding}")
                    # update document tracker db to contain new file
                    self.sql_document_tracker.insert_document_and_embedding(name=file, embedding=self.embedding)
            
            # process and upload the new files to vector store
            if new_knowledge:
                for knowledge in new_knowledge:
                    loader = PyPDFLoader(f"data/{knowledge}")
                    documents = loader.load_and_split()
                    split_documents = self.text_splitter.split_documents(documents)
                    self.add_to_chroma(split_documents)
            else:
                print('no new data to be added')
        except Exception as e:
            print(f"Error in VectorStoreManager.ingest_data: {e}")
        
    def add_to_chroma(self, chunks):
        try:
            vs = self.vector_store

            # Calculate Page IDs.
            chunks_with_ids = self.calculate_chunk_ids(chunks)

            # get documents in vector store
            existing_items = vs.get(include=[])  
            existing_ids = set(existing_items["ids"])
            print(f"Number of existing documents in vs: {len(existing_ids)}")

            # add documents that don't exist in the DB.
            new_chunks = []
            for chunk in chunks_with_ids:
                if chunk.metadata["id"] not in existing_ids:
                    new_chunks.append(chunk)

            # upload new document chunks to vector store
            if len(new_chunks):
                new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
                print(new_chunks)
                vs.add_documents(documents=new_chunks, ids=new_chunk_ids)
                print("successfully added new documents")
            else:
                print("No new documents to add")
        except Exception as e:
            print(f"Error in VectorStoreManager.add_to_chroma: {e}")

    def calculate_chunk_ids(self, chunks):
        """
        Generates unique IDs for document chunks based on their source and page number.
        Returns This will create IDs like "doc: POL011BA.pdf page: 6:2"
        Page Source : Page Number : Chunk Index
        """
        try:
            

            last_page_id = None
            current_chunk_index = 0

            # calculate ids for all chunks
            for chunk in chunks:
                chunk.metadata['embedding']=self.embedding
                source = chunk.metadata.get("source")[4:]
                page = chunk.metadata.get("page")
                current_page_id = f"doc: {source} page:{page}"

                # increment index of chunks with with the same page number
                current_chunk_index = current_chunk_index + 1 if last_page_id == current_page_id else 0

                # create chunk ID.
                chunk_id = f"{current_page_id}:{current_chunk_index}"
                # add ID to metadata
                chunk.metadata["id"] = chunk_id

                last_page_id = current_page_id
            return chunks
        except Exception as e:
            print(f"Error in VectorStoreManager.calculate_chunk_ids: {e}")
  
class SqlDb:
    def __init__(self) -> None:
        try:
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
            print(f"Error in SqlDb.__init__: {e}")

    def insert_document_and_embedding(self, name: str, embedding: str) -> None:
        try:
            with sqlite3.connect('db/knowledge_files_tracker2.db') as conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO documents (doc_title, embedding_name) VALUES (?, ?)', (name, embedding))
                conn.commit()
        except sqlite3.IntegrityError as i:
            print(f"sqlite3.IntegrityError in SqlDb.insert_document_and_embedding {e}")
        except Exception as e:
            print(f"Error in SqlDb.insert_document_and_embedding: {e}")

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
            print(f"Error in SqlDb.insert_question: {e}")

    def document_with_embedding_exists(self, doc_id: str, embedding: str) -> bool:
        try:
            with sqlite3.connect('db/knowledge_files_tracker2.db') as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1 FROM documents WHERE doc_title = ? AND embedding_name = ?', (doc_id, embedding))
                conn.commit()
            return bool(cursor.fetchone())
        except Exception as e:
            print(f"Error in SqlDb.document_with_embedding_exists: {e}")
            return False

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
            print(f"Error in SqlDb.get_question_by_doc_id: {e}")
            return None

    def doc_id_has_question(self, doc_id: str) -> bool:
        try:
            with sqlite3.connect('db/knowledge_files_tracker2.db') as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1 FROM qa_pairs WHERE doc_id = ? LIMIT 1', (doc_id,))
                result = cursor.fetchone()
                return result is not None
        except Exception as e:
            print(f"Error in SqlDb.doc_id_has_question: {e}")
            return False

    def get_all_entries(self, embedding: str) -> list:
        try:
            with sqlite3.connect('db/knowledge_files_tracker2.db') as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT doc_title FROM documents WHERE embedding_name = ?', (embedding,))
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error in SqlDb.get_all_entries: {e}")
            return []

    def delete_entry(self, doc_id: str, embedding: str) -> None:
        try:
            with sqlite3.connect('db/knowledge_files_tracker2.db') as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM documents WHERE doc_title = ? AND embedding_name = ?', (doc_id, embedding))
                conn.commit()
        except Exception as e:
            print(f"Error in SqlDb.delete_entry: {e}")

import time
import inquirer
from src.test_generator import TestQuestionGenerator
from src.model import Model
from src.vectorstore import VectorStoreManager, SqlDb
from src.pipeline import Pipeline
from src.schemas.question import Question
from enum import Enum
from langchain_cohere import CohereEmbeddings
from langchain_openai import OpenAIEmbeddings
from tabulate import tabulate

# Define available embeddings using an Enum
class Embedding(Enum):
    COHERE = CohereEmbeddings(model="embed-english-v3.0")
    OPENAI_SMALL = OpenAIEmbeddings(model="text-embedding-3-small")
    OPENAI_LARGE = OpenAIEmbeddings(model="text-embedding-3-large")

def main():
    # CLI prompts for selecting embedding and number of experiments
    questions = [
        inquirer.List(
            "embedding",
            message="Choose the embedding model to use:",
            choices=["COHERE", "OPENAI_SMALL", "OPENAI_LARGE"],
        ),
        inquirer.Text(
            "experiments",
            message="Enter the number of experiments to run:",
            default="10",
            validate=lambda _, x: x.isdigit() and int(x) > 0
        ),
    ]

    # Capture answers
    answers = inquirer.prompt(questions)

    # Convert selections to appropriate types
    selected_embedding = Embedding[answers["embedding"]].value
    num_experiments = int(answers["experiments"])

    # Initialize components
    llm = Model()
    RAG_pipeline = Pipeline(
        model=llm, 
        embedding_function=selected_embedding, 
        vector_store_manager=VectorStoreManager()
    )
    x = TestQuestionGenerator()
    v = VectorStoreManager()
    sql = SqlDb()
    a = x.pick_random_document()
    qa = x.generate_test_case(a)
    
    # Run the experiments
    results = []
    total_start_time = time.time()  # Start timing the total test
    success_count = 0
    total_iteration_time = 0

    for i in range(num_experiments):
        iteration_start_time = time.time()  # Start timing this iteration
        result = x.run_test_case(pipeline_to_test=RAG_pipeline, test_case=qa)
        results.append(result)

        if result:
            success_count += 1
        
        iteration_end_time = time.time()  # End timing this iteration
        iteration_duration = iteration_end_time - iteration_start_time
        total_iteration_time += iteration_duration
        print(f"Iteration {i+1} took {iteration_duration:.4f} seconds")

    total_end_time = time.time()  # End timing the total test
    total_duration = total_end_time - total_start_time

    # Calculate metrics
    success_rate = success_count / num_experiments * 100
    average_iteration_time = total_iteration_time / num_experiments

    # Prepare data for tabulation
    table = [
        ["Total Experiments", num_experiments],
        ["Successes", success_count],
        ["Failures", num_experiments - success_count],
        ["Success Rate (%)", f"{success_rate:.2f}"],
        ["Total Duration (s)", f"{total_duration:.4f}"],
        ["Average Iteration Time (s)", f"{average_iteration_time:.4f}"]
    ]

    # Print the results in a tabulated format
    print("\nTest Results Summary")
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))

if __name__ == "__main__":
    main()
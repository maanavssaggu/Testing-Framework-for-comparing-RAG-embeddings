# Testing Framework for Comparing RAG Embeddings

This project provides a testing framework to evaluate and compare the performance of different embedding models. The models currently supported for evaluation are:

1. **Cohere**: `embed-english-v3.0`
2. **OpenAI Small**: `text-embedding-3-small`
3. **OpenAI Large**: `text-embedding-3-large`

The framework works by determining whether a selected embedding model can accurately retrieve a specific document in response to a question generated about it. The process involves the following steps:

1. **Document Selection**: A random document is selected from the knowledge store.

2. **Test Case Generation**:
   - The framework checks if a test case already exists for the selected document.
   - If no test case exists, GPT-4o generates a True/False question about the document.

3. **Document Retrieval**:
   - The embedding model under evaluation is integrated into a generative pipeline.
   - The pipeline retrieves the top 5 (`k=5`) documents that are most relevant to the generated question.

4. **Evaluation**:
   - If the originally selected document is among the 5 retrieved documents, the test is considered passed.
   - The accuracy of the embedding model is assessed based on the number of successful retrievals out of `n` test cases.

This test can be run `n` times, and the final accuracy is printed at the bottom.

### Figures below describe how the model works:

![image](https://github.com/user-attachments/assets/04850804-c992-42e3-ac70-2b42e4f32198)

![image](https://github.com/user-attachments/assets/8abf04ea-d729-4219-b688-0ab48088b16a)

## Features

- Persistent storage of generated questions
- CLI interface using inquirer

## Usage/Examples

```terminal
python -m src.main
[?] Choose the embedding model to use:: 
   COHERE
   OPENAI_SMALL
 > OPENAI_LARGE

[?] Enter the number of experiments to run:: 10
Iteration 10 took 3.9256 seconds

Test Results Summary
+----------------------------+----------+
| Metric                     |    Value |
+============================+==========+
| Total Experiments          |  10      |
+----------------------------+----------+
| Successes                  |  10      |
+----------------------------+----------+
| Failures                   |   0      |
+----------------------------+----------+
| Success Rate (%)           | 100      |
+----------------------------+----------+
| Total Duration (s)         |  42.0302 |
+----------------------------+----------+
| Average Iteration Time (s) |   4.203  |
+----------------------------+----------+

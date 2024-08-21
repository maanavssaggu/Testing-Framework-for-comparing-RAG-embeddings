
# Testing Framework for comparing RAG embeddings

This project allows you to compare the following embedding model using a self-made testing framework: 
1. COHERE = embed-english-v3.0
2. OPENAI_SMALL = text-embedding-3-small
3. OPENAI_LARGE = text-embedding-3-large. 

The framework works by seeing whether or not the pipeline can retireve the specific requried document for a question contrieved by the testing framework. The framework creates these questions by picking a random document from the knowledge store to then ask gpt-4o to generate True or False question about it. 

Then the embedding model you choose to asses will be used in the generative pipeline and will retrieve 5 documents (k=5) and if the particular document is among the 5 it will then pass.   

This test can be run n times, and the final accuracy is printed at the bottom. 

Figures below describe how the model works: 

![image](https://github.com/user-attachments/assets/a68a09a9-30e9-492c-b1f5-ed1a716ef771)


![image](https://github.com/user-attachments/assets/8abf04ea-d729-4219-b688-0ab48088b16a)

# Testing Framework for comparing RAG embeddings

This project allows you to compare any embedding model (provided it is supported by langchain) by testing the accuracy of its top 5 pulled documents. 




## Features

- persistent storage of generated questions
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

```

## Terminal output of running the framework
<img width="674" alt="image" src="https://github.com/user-attachments/assets/813f5992-7c5f-4297-a61a-6152a8a752cb">




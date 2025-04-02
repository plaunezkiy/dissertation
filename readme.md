## Domain-specific QA with LLMs and Knowledge Graphs
This repository contains code for data processing, experimentation framework and notebooks processing, and displaying final results.

It has several modules and a library with locally used tools

Folder breakdown:
- `code/0_data` has initial dataset loading code that generates a 1000 entries subset for each of the 3 selected datasets 
- `code/1_baseline` has the code for obtaining baseline results. Namely bline + bline2, and kb-path (for perfect shortest path on the graph)
- `code/2_kg_inference` has the code for each of the attempted heuristics. Due to how experimentation was done, `kb1` has single fixed-depth of 1, `kbN` identifies hop number for each question and records that, `kbN+1` is for n $\pm1$ around each hop number for every question, `ToG` has attempts at recreating the original ToG implementation and the link predictor optimised heuristic. `results` has all tables and visualisations for all of the methods.
- `code/3_link_prediction` has some experimental code for smaller LLM as a link predictor, which is where the idea was developed
- `code/4_kb_app` has early attempts at building a graphical user interface for navigating and querying actual technical documentation.
- `code/utils` has all of the custom modules and functions developed for this project:
    1. `utils/graph.__init__.py` contains knowledge graph construction and embedding functionality for each of the 3 datasets
    2. `utils/chain.py` contains Langchain-based graph traversal module with the 2 relevance heuristics (BM25, S-BERT)
    3. `utils/tog_lp.py` contains Langchain-based graph traversal module for the ToG-LP modification of the algorithm
    4. `utils/llm` contains langchain compatible modules for the custom `Mistral7B` and `Qwen2.5-0.5B` LLM modules modified for parallel inference.
    5. `utils/plot.py` contains code for truncated axis plotting
    6. `utils/preprocessing.py` has all of the text preprocessing functionality - tokenization and stemming.
    7. `utils/prompt.py` contains all of the variable-adaptable prompts to be used with LLMs
    8. `utils/scores.py` contains the functions for calculating Exact Match (EM) and F1 scores used for evaluation.
    9. `utils/evaluation.py` has the logic for loading all test sets, has a modifiable list of result sets for each of the datasets. The results are loaded, the scores are computed and printed as a table formatted for readablity.

- `docker-compose.yml` has a config for creating a docker container with a GPU, and a notebook exposing ports to be used on a remote cluster
- `Dockerfile` has a system setup config for the container with all dependencies and python version.
- `requirements.txt` contains a list of dependencies (python libraries) needed for the project
- `misc/` has some experimental code with different models
- `datasets/` contains condensed 1,000 items subsets and contain `results/` subfolder that has .csv files with the results of experiments
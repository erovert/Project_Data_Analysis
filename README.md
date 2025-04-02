# Project_Data_Analysis

This repository contains a data analysis project that uses NLP techniques to analyze a collection of customer complaints about Comcast. The goal is to extract the most prevalent topics from the unstructured text data and evaluate the results using topic modeling techniques.

## Project Overview

In this project:
- **Preprocessed** the raw complaint texts using SpaCy for lemmatization, stop-word removal, and tokenization.
- **Vectorized** the clean texts using two techniques: TF-IDF and Word2Vec-based embeddings.
- **Extracted topics** using two methods: Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF).
- **Evaluated** the quality of the topics by computing coherence scores.

## Repository Contents

- **`data_analysis_script.py`**: The main Python script containing the complete analysis pipeline.
- **`comcast_consumeraffairs_complaints.csv`**: The dataset of customer complaints.
- **`README.md`**: This file.

## Dependencies

The project requires the following Python libraries:
- pandas
- spacy
- tqdm
- numpy
- scikit-learn
- gensim
- matplotlib (optional, if you add visualization)
- scikit-learn's KMeans for clustering

To install the dependencies, run:

```
pip install pandas spacy tqdm numpy scikit-learn gensim
```
```
python -m spacy download en_core_web_sm
```

## How to Run the Project
Clone the repository:
```
git clone https://github.com/erovert/Project_Data_Analysis.git
```

```
cd Project_Data_Analysis
```
Run the script:
```
python data_analysis_script.py
```
This script will:


Preprocess the text data.

Build the TF-IDF representation and extract topics using LDA and NMF.

Build Word2Vec document embeddings, perform clustering, and extract topics from a bag-of-clusters representation.

Print coherence scores and sample topics in the console.

Results and Reflection

The TF-IDF pipeline (especially with NMF) produced more coherent and interpretable topics (with coherence scores around 0.51) compared to the Word2Vec-based approach. These results can be further improved by tuning hyperparameters, such as the number of topics, and refining the preprocessing steps.

Future Improvements

Potential future improvements include:

Tuning model parameters (e.g., the number of topics, NMF iterations) to enhance coherence.

Incorporating domain-specific stop words.

Adding visualizations for better insights.

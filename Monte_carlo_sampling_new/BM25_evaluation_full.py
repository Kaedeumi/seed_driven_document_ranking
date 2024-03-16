import pandas as pd
import math
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from keywordExtraction_and_MC_sampling import vectorizer, truly_positive_docs, test_path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json


# Function to preprocess text: tokenize, remove stopwords, stem/lemmatize
lemmatizer = WordNetLemmatizer()
truly_positive_text_orders = [doc['order'] for doc in truly_positive_docs]

def read_test_files(test_path):
    # Initialize a list to store the data
    data = []

    # Open the file and read each line
    with open(test_path, 'r') as file:
        for line in file:
            # Convert the line from JSON format to a Python dictionary
            entry = json.loads(line)
            # Extract the 'order' and 'abstract' fields
            data.append({
                'Document Order': entry.get('order', 'NA'),  # Use 'NA' if 'order' is not found
                'Abstract': entry.get('abstract', 'NA')  # Use 'NA' if 'abstract' is not found
            })

    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file (optional)
    df.to_csv('extracted_data.csv', index=False)

    # Print the first few rows of the DataFrame to check
    print(df.head())

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stopwords.words('english')]
    return ' '.join(tokens)

# Function to calculate term frequency in a document
def calculate_tf(term, document):
    return document.count(term)

# Function to calculate inverse document frequency
def calculate_idf(term, documents):
    num_docs_with_term = sum(1 for doc in documents if term in doc)
    return math.log((len(documents) + 1) / (num_docs_with_term + 1)) + 1

# Function to calculate BM25 for a single document
def bm25_score(document, query_terms, idf_values, avg_doc_length, k1=1.5, b=0.75):
    score = 0
    for term in query_terms:
        if term in idf_values:
            tf = calculate_tf(term, document)
            idf = idf_values[term]
            score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (len(document) / avg_doc_length))))
    return score

def bm25_score_calculation():
    '''
    This method calculates the BM25 scores of each document in the testing set with respect to the
    feature vector Q.

    :param
    :return: list of lists of truly_pos and LP positions
    '''

    # Read the CSV file
    df = pd.read_csv(f'extracted_data.csv')

    # Read the orders of the retrieved documents
    # Load the CSV file into a DataFrame

    # Extract the 'Document Order' column and convert it to a list
    document_order_list = df['Document Order'].tolist()

    # Now document_order_list contains all the 'Document Order' values from the CSV
    print(document_order_list)

    print(truly_positive_text_orders)

    # Assuming feature vector Q is already defined and preprocessed
    feature_vector_Q = vectorizer  # This should be a list of preprocessed terms

    # Preprocess abstracts from the CSV file
    df['processed_abstract'] = df['Abstract'].apply(preprocess_text)

    # Calculate IDF for each term in the feature vector across all abstracts
    documents = df['processed_abstract'].tolist()
    # print(documents)

    avg_doc_length = sum(len(doc) for doc in documents) / len(documents)
    # Accessing feature names from the TF-IDF matrix
    feature_names = feature_vector_Q.get_feature_names_out()

    # Calculate IDF for each term in the feature vector across all abstracts
    idf_values = {}
    for term in tqdm(feature_names, desc='Calculating IDF'):
        idf_values[term] = calculate_idf(term, documents)

    # Preprocess the feature vector
    preprocessed_feature_vector_Q = [preprocess_text(term) for term in feature_names]

    # Calculate BM25 scores for each abstract
    df['bm25_score'] = df['processed_abstract'].apply(
        lambda doc: bm25_score(doc, preprocessed_feature_vector_Q, idf_values, avg_doc_length)
    )

    # Output the DataFrame with BM25 scores
    df.to_csv(f'extracted_data.csv', index=False)

    # Add a column representing the normalized scores based on the files above
    # Load the data into a DataFrame
    df_append = pd.read_csv(f'extracted_data.csv')

    # Calculate the normalized BM25 score
    df_append['normalized_bm25_score'] = (df_append['bm25_score'] - df_append['bm25_score'].min()) / (
            df_append['bm25_score'].max() - df_append['bm25_score'].min())

    # Save the updated DataFrame back to a CSV
    df_append.to_csv(f'extracted_data.csv', index=False)


read_test_files(test_path)
bm25_score_calculation()
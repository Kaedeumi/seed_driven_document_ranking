"""
    This file preprocesses the dataset, chooses the seed corpus,extracts the keywords
    based on the tfidf weighting, and outputs the top 10 keywords to a csv file.
    This script should be run first.

    Steps involved in the original article:
    1. Keyword Extraction: Identifying a set of seed documents from LP set and extract keyword from it
    2. Database Sampling: Monte Carlo sampling from the list of extracted keywords, followed by constructing
    queries from the probability distributions. The best parameters and keywords are then returned for the
    next steps.
"""
import json
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from elasticsearch import Elasticsearch
import matplotlib.pyplot as plt
from collections import OrderedDict


# Load the dataset
positive_texts = []     # this is a list of strings
labeled_positive = []   # this is a list of dictionaries, whose purpose is to access the orders
all_papers = []
# Specify the input and output file paths
input_file_path = '../data/pubmed-dse/L20/D000328.D008875.D015658/train.jsonl'
ordered_input_file_path = '../data/pubmed-dse/L20/D000328.D008875.D015658/ordered_train.jsonl'
output_file_path = 'labeledpositives.jsonl'
seed_output_file_path = 'seeds.txt'

def reading_all_papers(ordered_input_file_path):
    with open(ordered_input_file_path, 'r') as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            all_papers.append({'Order': i + 1, 'Abstract': data['abstract']})
    return all_papers

all_papers = reading_all_papers(ordered_input_file_path)

def reading_LP_as_dict(ordered_input_file_path):
    lp_order_list = []
    with open(ordered_input_file_path, 'r') as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            if data['label_true'] == 'positive/labeled':
                # Append a dictionary with 'Order' and 'Abstract'
                labeled_positive.append({'Order': i + 1, 'Abstract': data['abstract']})
                lp_order_list.append(data['order'])
    return labeled_positive, lp_order_list

def reading_LP_as_strs(input_file_path):
    with open(input_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if data['label_true'] == 'positive/labeled':  # Only add texts from positive examples
                positive_texts.append(data['abstract'])
    return positive_texts

import json

labeled_positive, lp_order_list = reading_LP_as_dict(ordered_input_file_path)
positive_texts = reading_LP_as_strs(input_file_path)

# to-be-done: modify this function such that the seed documents returned are those for which the order and abstract fields are both kept in track
def get_seed_documents(labeled_positive, L):
    # Random selection of L instances from 'seed_documents'
    random.seed(42)  # Set a random seed for reproducibility

    # Randomly select L documents from the labeled positives
    selected_documents = random.sample(labeled_positive, L)

    # Transform selected documents into the desired format
    seed_documents = [{'order': i + 1, 'abstract': text} for i, text in enumerate(selected_documents)]

    return seed_documents

seed_documents = get_seed_documents(labeled_positive, 20)
print(seed_documents)

# Function to read .jsonl file and write documents to a separate file
def print_jsonl_to_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file, \
            open(output_file_path, 'w', encoding='utf-8') as out_file:
        for line_number, line in enumerate(file, start=1):
            # Convert string to a JSON object
            json_obj = json.loads(line)

            # Write the JSON object to the output file
            out_file.write(f"Line {line_number}: {json.dumps(json_obj)}\n")

            # Optionally, you could limit the number of lines to avoid a very large output file
            # if line_number == 10:
            #     break


def print_seeds_to_file(seeds, seed_output_file_path):
    """
    Writes a list of dictionaries, each with an 'order' and an 'abstract', to a specified file.

    Parameters:
    - seeds: List[Dict[str, Union[int, str]]]. A list of dictionaries, each containing 'order' and 'abstract' keys.
    - seed_output_file_path: str. The path to the file where the dictionaries will be written.

    Returns:
    None
    """
    seed_order_list = []
    with open(seed_output_file_path, 'w', encoding='utf-8') as file:
        for seed in seeds:
            order = seed['order']
            abstract = seed['abstract']
            file.write(f'No. {order}: {abstract}\n')
            seed_order_list.append(abstract['Order'])
    return seed_order_list

# Call the function to write the contents of the JSON lines file to a separate file
print_jsonl_to_file(input_file_path, output_file_path)
seed_order_list = print_seeds_to_file(seed_documents,seed_output_file_path)

# Assuming you have already downloaded the necessary NLTK resources
# Preprocess the text: here we tokenize and lemmatize the text
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stopwords.words('english')]
    return ' '.join(tokens)

# Preprocess the positive texts
seed_texts = [doc['abstract'] for doc in seed_documents]
preprocessed_texts = [preprocess_text(doc['Abstract']) for doc in seed_texts] # seed_texts is still a dictionary !!!!!


# Calculate TF-IDF weights
vectorizer = TfidfVectorizer(use_idf=True,sublinear_tf=False)
tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)

# Get feature names to use as an index to sort the TF-IDF scores
feature_names = vectorizer.get_feature_names_out()

def get_top_keywords(tfidf_matrix, feature_names, top_n=10):
    # Sum the TF-IDF scores for each term across all documents
    summed_tfidf = np.array(tfidf_matrix.sum(axis=0)).flatten()

    # Get the indices of the top N terms
    top_indices = np.argsort(summed_tfidf)[-top_n:]

    # Get the top N terms and their summed TF-IDF scores
    top_terms = [(feature_names[index], summed_tfidf[index]) for index in reversed(top_indices)]

    return top_terms

# Use this function with your TF-IDF matrix and feature names
top_keywords = get_top_keywords(tfidf_matrix, feature_names)

# Create a DataFrame for the top terms
df_top_terms = pd.DataFrame(top_keywords, columns=['Term', 'TF-IDF Score'])

# Sort the DataFrame in descending order of TF-IDF scores
df_top_terms = df_top_terms.sort_values(by='TF-IDF Score', ascending=False)

# Output the DataFrame to a CSV file
df_top_terms.to_csv('top_keywords.csv', index=False)

# Print the DataFrame
print(df_top_terms)

# Monte-Carlo Sampling to generate queries =======================================================================
# Convert the TF-IDF matrix to a dense format
dense_tfidf = tfidf_matrix.todense()

# Sum the TF-IDF scores for each term across all documents
summed_tfidf = np.array(np.sum(dense_tfidf, axis=0)).flatten()

# Rank terms based on their TF-IDF weights
sorted_term_indices = np.argsort(summed_tfidf)[::-1]  # Indices of terms in descending order of score
sorted_terms = feature_names[sorted_term_indices]  # Terms in descending order of score
# Compute probabilities for Monte Carlo sampling
probabilities = summed_tfidf[sorted_term_indices] / np.sum(summed_tfidf)

# Create a DataFrame using the terms and their probabilities
df_terms_prob = pd.DataFrame({
    'Term': sorted_terms,
    'Probability': probabilities
})

# Assuming df_terms_prob is your existing DataFrame
# Reset the index to make the current index into a column
df_terms_prob.reset_index(inplace=True)

# Now rename the new column to 'order' and start the count at 1
df_terms_prob.rename(columns={'index': 'order'}, inplace=True)
df_terms_prob['order'] += 1

# If you want to sort the DataFrame alphabetically by the terms
df_terms_prob.sort_values(by='Term', inplace=True)

# Reset the 'order' column to reflect the new sorting
df_terms_prob['order'] = range(1, len(df_terms_prob) + 1)
# Export the DataFrame to a CSV file
output_file_path = 'terms_with_probabilities.csv'
df_terms_prob.to_csv(output_file_path, index=False)

# Print a message indicating completion
print(f'Terms and probabilities have been written to {output_file_path}')

import random
from tqdm import tqdm  # tqdm is a library in Python that provides a progress bar for loops

def monte_carlo_sampling_with_fixed_params(N_MC, N_kw):
    # Fixed Parameters for Monte Carlo sampling -----------------------------------------------------------------------
    N_MC = 1000  # Number of Monte Carlo iterations (between 200 and 1000)
    N_it = 2000  # Upper limit for the number of documents registered in each iteration (Scopus API limit)
    N_kw = 5    # Number of keywords included in the sampling (default value)

    # Function to perform Monte Carlo sampling to generate a single query
    def monte_carlo_sampling(sorted_terms, probabilities, N_kw=N_kw):
        # Randomly choose terms based on their TF-IDF weight-derived probabilities
        sampled_keywords = random.choices(sorted_terms, weights=probabilities, k=N_kw)
        return sampled_keywords

    # Perform the Monte Carlo sampling with a progress bar
    document_appearance_count = {}
    for _ in tqdm(range(N_MC), desc='Monte Carlo Sampling', unit='iteration'):
        sampled_keywords = monte_carlo_sampling(sorted_terms, probabilities)
        # print(sampled_keywords)
        # Register the documents based on sampled keywords
        for doc in preprocessed_texts:
            if all(keyword in doc for keyword in sampled_keywords):
                if doc in document_appearance_count:
                    document_appearance_count[doc] += 1
                else:
                    document_appearance_count[doc] = 1

    # Compute document frequency (DF) for each document
    document_frequency = {doc: count / N_MC for doc, count in document_appearance_count.items()}

    # Sort the documents by their document frequency (DF)
    sorted_documents = sorted(document_frequency.items(), key=lambda item: item[1], reverse=True)

    # Print the sorted documents and their frequencies
    i = 0
    for doc, freq in sorted_documents:
        i += 1
        print(f'Document no. {i}: {doc}, Frequency: {freq}')

# Parameter sweep procedure --------------------------------------------------------------------------
from itertools import product


# Best performance tracking
from itertools import product
import random
from tqdm import tqdm

# Assuming these variables are defined somewhere in your code

# Replace with your actual username and password
username = 'yang'
password = 'theno1ofdmt'

# Connect to Elasticsearch with authentication details
es = Elasticsearch(
    hosts=["http://localhost:9200"],
    http_auth=(username, password)
)

# Define the custom analyzer using built-in filters and tokenizers
custom_analyzer = {
    "settings": {
        "analysis": {
            "analyzer": {
                "custom_english_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "english_stop",  # Stop words filter
                        "english_keywords",  # Keywords filter
                        "english_stemmer",  # Stemmer filter
                        "english_possessive_stemmer"  # Stemmer filter for possessive words
                    ]
                }
            },
            "filter": {
                "english_stop": {
                    "type": "stop",
                    "stopwords": "_english_"
                },
                "english_keywords": {
                    "type": "keyword_marker",
                    "keywords": []  # List of terms you do not want to be stemmed.
                },
                "english_stemmer": {
                    "type": "stemmer",
                    "language": "english"
                },
                "english_possessive_stemmer": {
                    "type": "stemmer",
                    "language": "possessive_english"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "abstract": {
                "type": "text",
                "analyzer": "custom_english_analyzer"
            },
            "label": {
                "type": "keyword"
            }
        }
    }
}

# Define the index where you want to store your documents
index_name = 'database_sampling2'

def index_documents(jsonl_file_path, index_name):
    i = 0
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            i += 1
            doc = json.loads(line)
            res = es.index(index=index_name, document=doc)
            print('indexing document', i)
            print(res['result'])  # Prints whether the indexing operation was successful ('created')

# index_documents(r'../../data/pubmed-dse/L20/D000328.D008875.D015658/ordered_train.jsonl', index_name)
# Define the Monte Carlo sampling function inside the loop if it uses N_kw
def monte_carlo_sampling(sorted_terms, probabilities, N_kw):
    sampled_keywords = random.choices(sorted_terms, weights=probabilities, k=N_kw)
    return sampled_keywords

def send_query_to_database(keywords,n_kw):
        # Assuming 'keywords' is a list of strings ['keyword1', 'keyword2', ..., 'keywordN']
        # Construct the query. The following is a simple match query, you might need to adjust it to your needs

        min_should_match = int(n_kw * 0.4)
        query = {
            "query": {
                "bool": {
                    "should": [{"match": {"abstract": keyword}} for keyword in keywords],
                    "minimum_should_match": min_should_match  # This means at least one of the clauses should match
                    # This assumes your documents have a 'content' field to search within
                }
            }
        }

        # Send the query to Elasticsearch
        response = es.search(index=index_name,
                             body=query)  # Replace 'your_index_name' with the name of your index

        # Extract the documents from the response
        # The following line extracts the '_source' of Elasticsearch hits which contain your actual documents
        documents = [hit["_source"] for hit in response['hits']['hits']]

        return documents

# Initialize the vectorizer
vectorizer2 = TfidfVectorizer(use_idf=True, sublinear_tf=False)
vectorizer2.fit(positive_texts)

# This method is only run after the MC iterations
def printing_sorted_docs_with_frequencies(sorted_documents,n_mc,n_kw):
    # Print the sorted documents with their frequencies
    # for doc, freq in sorted_documents:
    #     print(f'Document Order: {doc}, Frequency: {freq:.4f}, Occurrences {freq * n_mc}')

    # Optionally, write the sorted documents with their frequencies to a CSV file
    import csv

    # Define the CSV file path
    csv_output_file_path = f'parameter_sweeping/document_frequencies_nmc{n_mc}_nkw{n_kw}.csv'

    # Write the document frequencies to the CSV file
    with (open(csv_output_file_path, 'w', newline='', encoding='utf-8') as csvfile):
        csvwriter = csv.writer(csvfile)
        # Writing header with an additional 'Raw DF' column
        csvwriter.writerow(['Document Order', 'Normalized Frequency', 'Raw DF', 'Abstract'])

        for doc, count in sorted_documents:
            # Calculate normalized frequency
            normalized_freq = count * n_mc

            # get the abstract field for the corresponding order
            abstract_text = ''
            # Iterate through the seed_documents list
            for docs in all_papers:
                # Check if the 'order' field matches the target_order
                if docs['Order'] == doc:
                    # Access the 'Abstract' field
                    abstract_text = docs['Abstract']
                    # print(abstract_text)
            # Write document order, normalized frequency, and raw count to CSV
            csvwriter.writerow([doc, count, f'{normalized_freq:.4f}',abstract_text])


    # Function to calculate performance metrics

def calculate_performance_metrics(retrieved_documents, seed_documents):
    '''
    In this method, it's implicitly assumed that the retrieved_documents and seed_documents are both list of strings.

    :param retrieved_documents: list of strings that represent the documents retrieved after each query.
    note that here we denote 'retrieved_documents' as the whole set of retrieved documents after n_mc iterations.
    :param seed_documents: list of strings that represent the set of seed documents defined.
    :return: seed recall and precision for each query
    '''

    retrieved_set = set(retrieved_documents)
    seed_set = set(seed_documents)
    true_positive_set = retrieved_set & seed_set

    # Calculate seed recall
    true_positives = len(set(retrieved_documents) & set(seed_documents))
    recall = true_positives / len(seed_documents) if seed_documents else 0

    # Calculate average seed precision
    precision = true_positives / len(retrieved_documents) if retrieved_documents else 0

    return recall, precision, true_positive_set

# New function to write metrics and sets to a text file
def print_metrics_and_sets(recall, precision, true_positives, seed_documents, retrieved_documents, n_mc, n_kw):
    # Define the text file path
    txt_output_file_path = f'iteration_metrics.txt'

    # Write the metrics and sets to the text file
    with open(txt_output_file_path, 'a', encoding='utf-8') as txtfile:  # 'a' for append mode
        # Write the parameter combination for clarity
        txtfile.write(f'Parameter combination: N_MC={n_mc}, N_KW={n_kw}\n')
        txtfile.write(f'Recall: {recall:.4f}\n')
        txtfile.write(f'Precision: {precision:.4f}\n')
        txtfile.write(f"True Positives (Order numbers): {sorted(true_positives)}\n")
        txtfile.write(f"Seed Documents (Order numbers): {sorted(seed_documents)}\n")
        txtfile.write(f"Retrieved Documents (Order numbers): {sorted(retrieved_documents)}\n")
        txtfile.write('\n')  # Add a newline for separation between entries

def iteration_performance_visualization(precision_dict, recall_dict):
    # Sort the keys to ensure the bars are in order
    parameter_combinations = sorted(precision_dict.keys())

    # Extract the precision and recall values in the same order
    precisions = [precision_dict[param] for param in parameter_combinations]
    recalls = [recall_dict[param] for param in parameter_combinations]

    # Define the scaling factor for precision bars
    scaling_factor = 0.05

    # Plotting the bar chart
    x = range(len(parameter_combinations))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figsize parameter as needed
    # Convert x to a list if it's a range and perform the arithmetic operation
    # Adjust the x-coordinates for the precision bars to be shifted to the left
    adjusted_x_precision = [xi - width / 2 for xi in list(x)]
    # Adjust the x-coordinates for the recall bars to be shifted to the right
    adjusted_x_recall = [xi + width / 2 for xi in list(x)]

    # Plot the scaled precision bars
    rects1 = ax.bar(adjusted_x_precision, [p / scaling_factor for p in precisions], width, label='Precision (scaled)')

    # Recall bars remain unchanged
    rects2 = ax.bar(adjusted_x_recall, recalls, width, label='Recall')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('Parameter Combinations (n_mc, n_kw)')
    ax.set_ylabel('Scores')
    ax.set_title('Precision and Recall by Parameter Combination')
    ax.set_xticks(x)
    ax.set_xticklabels(parameter_combinations)
    ax.legend()

    # Function to attach a text label above each bar, displaying its height
    def autolabel(rects, is_scaled=False, scaling_factor=1):
        for rect in rects:
            height = rect.get_height()
            if is_scaled:
                # If the bars are scaled, the label should show the actual value (not scaled)
                label_value = height * scaling_factor
            else:
                label_value = height

            # Apply a consistent offset from the top of the bar
            offset = 3  # pixels
            ax.annotate(f'{label_value:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, offset),  # Offset from the top of the bar
                        textcoords="offset points",
                        ha='center', va='bottom')

    # When calling autolabel for precision bars, indicate that they are scaled
    autolabel(rects1, is_scaled=True, scaling_factor=scaling_factor)

    # For recall bars, no scaling is applied so the function is called normally
    autolabel(rects2)

    # Customize x-axis labels
    # plt.xticks(rotation=30, ha='right')  # Rotate the labels and align them to the right
    plt.tight_layout()  # Adjust the padding between and around subplots.
    # Other plotting code remains unchanged
    # Show the plot
    plt.show()

def monte_carlo_with_parameter_sweeping():
    # Define the range of values for N_MC and N_kw to test
    n_mc_values = [200, 600, 1000]  # Example values
    n_kw_values = [5, 10, 15]  # Example values

    best_performance = None
    best_params = None
    best_keywords = None

    output_log_file_path = 'output_logs.txt'
    document_sets = []

    precision_dict = {}
    recall_dict = {}

    # output data holds the displays regarding the performance for each reason of the loop (precision, recall..)
    output_data = []
    with (open(output_log_file_path, 'w', encoding='utf-8') as f):  # Open the file in append mode
        # Parameter sweep
        # Parameter sweep
        for n_mc, n_kw in product(n_mc_values, n_kw_values):
            print(f'\nIteration of N_MC = {n_mc}, N_kw = {n_kw} =====================================')
            f.write(f'\nIteration of N_MC = {n_mc}, N_kw = {n_kw} ==============================================================================================\n')

            # Perform the Monte Carlo sampling
            document_appearance_count = {}
            all_sampled_keywords = []  # Keep track of all sampled keywords for this parameter set

            iteration = 0
            for _ in tqdm(range(n_mc), desc='Monte Carlo Sampling', unit='iteration'):
                iteration += 1
                # Here, we assume `monte_carlo_sampling` is a function that performs the sampling and returns the sampled keywords
                sampled_keywords = monte_carlo_sampling(sorted_terms, probabilities, n_kw)

                print(f'iteration {iteration}/{n_mc} : {sampled_keywords}')
                f.write(f'iteration {iteration}/{n_mc} : {sampled_keywords}\n')

                # Query the database with the sampled keywords, and calculate similarity for each iteration
                retrieved_documents = send_query_to_database(sampled_keywords,n_kw)
                document_sets.append(retrieved_documents)
                print(retrieved_documents)

                # Process the retrieved documents as required

                i = 0
                for doc in retrieved_documents:
                    i += 1
                    print(f'=> {i}th document retrieved:',doc['order'], doc['title'],doc['abstract'])
                    f.write(
                        f'=> {i}th document retrieved: Order: {doc["order"]}, Title: {doc["title"]}, Abstract: {doc["abstract"]}\n')

                # Depending on your implementation, you might want to keep track of document appearances
                for doc in retrieved_documents:
                    doc_id = doc['order']
                    document_appearance_count[doc_id] = document_appearance_count.get(doc_id, 0) + 1

                all_sampled_keywords.append(sampled_keywords)  # Store sampled keywords for later analysis

            # Register the documents based on sampled keywords
            for doc in preprocessed_texts:
                if all(keyword in doc for keyword in sampled_keywords):
                    if doc in document_appearance_count:
                        document_appearance_count[doc] += 1
                    else:
                        document_appearance_count[doc] = 1

            # Compute document frequency (DF) for each document
            document_frequency = {doc: count / n_mc for doc, count in document_appearance_count.items()}

            # After performing the sampling, compute the performance
            sorted_documents = sorted(document_frequency.items(), key=lambda item: item[1], reverse=True)

            # printing the document frequencies for each parameter iteration cycle out
            printing_sorted_docs_with_frequencies(sorted_documents,n_mc,n_kw)

            # evaluating the performance to judge the best iteration among the 9
            # Assuming retrieved_documents is a list of dictionaries with a unique 'doc_id' key
            # Flatten the list of lists of dictionaries into a single list of dictionaries
            flattened_document_sets = [doc for sublist in sorted_documents for doc in sublist]
            # Extract the 'order' entries from each dictionary
            orders_of_retvd_docs = flattened_document_sets
            # Removing duplicates
            # Assuming orders_of_retvd_docs is your list with potential duplicates
            orders_of_retvd_docs = list(set(orders_of_retvd_docs))
            # This will preserve the order of the first occurrence of each element
            orders_of_retvd_docs = list(OrderedDict.fromkeys(orders_of_retvd_docs))

            # Now orders_of_retvd_docs contains only unique elements

            # Extract the 'order' entries from each dictionary
            unique_seed_docs = [doc['abstract']['Order'] for doc in seed_documents if 'Order' in doc['abstract']]

            performance = calculate_performance_metrics(orders_of_retvd_docs, unique_seed_docs)
            # after calculating the performance metrics:
            recall, precision, true_positives_set = calculate_performance_metrics(orders_of_retvd_docs, unique_seed_docs)

            # write them to the output file to summarize its performance
            output_data.append(f"Parameter combination: N_MC={n_mc}, N_KW={n_kw}")
            output_data.append(f"Recall: {recall:.4f}")
            output_data.append(f"Precision: {precision:.4f}")
            output_data.append(f"True Positives (Order numbers): {sorted(true_positives_set)}")
            output_data.append(f"Seed Documents (Order numbers): {sorted(unique_seed_docs)}")
            output_data.append(f"Retrieved Documents (Order numbers): {sorted(orders_of_retvd_docs)}\n")
            output_data.append("")  # Add an empty string for a newline separator between entries

            # Then call the new function to print the metrics and sets to a text file
            # print_metrics_and_sets(recall, precision, true_positives_set, unique_seed_docs, orders_of_retvd_docs, n_mc, n_kw)
            # Store the metrics in the dictionaries
            precision_dict[(n_mc, n_kw)] = precision
            recall_dict[(n_mc, n_kw)] = recall

            # call the visualization function to plot the barchart
            iteration_performance_visualization(precision_dict,recall_dict)

            # Check if the current performance is the best
            if best_performance is None or performance > best_performance:
                best_performance = performance
                best_params = {'N_MC': n_mc, 'N_kw': n_kw}
                # The best keywords are the ones from the iteration with the best performance
                best_keywords = all_sampled_keywords

    # Define the file path where the performance metrics will be saved
    performance_metrics_file_path = 'performance_metrics.txt'

    # Write the performance metrics to the text file
    with open(performance_metrics_file_path, 'w', encoding='utf-8') as file:
        file.write("Performance metrics for each combination of n_mc and n_kw:\n")
        file.write("{:<10} {:<10} {:<10} {:<10}\n".format("n_mc", "n_kw", "Precision", "Recall"))
        for params, precision in precision_dict.items():
            recall = recall_dict[params]
            file.write("{:<10} {:<10} {:<10.4f} {:<10.4f}\n".format(params[0], params[1], precision, recall))

    # Output the best performance and parameters
    print(f'Best Performance: {best_performance}')
    print(f'Best Parameters: {best_params}')
    # print(f'Best Keyword: {best_keywords}') # This will overwhelm the output
    import pandas as pd

    # Assume best_keywords is an array of sub-arrays containing string objects
    # Convert the array of sub-arrays into a DataFrame
    df_best_keywords = pd.DataFrame(best_keywords)

    # # Use a column header if you like, for example: ['Keyword 1', 'Keyword 2']
    # df_best_keywords = pd.DataFrame(best_keywords, columns=['Keyword 1', 'Keyword 2','Keyword 3','Keyword 4','Keyword 5',
    #                                                         'Keyword 6','Keyword 7','Keyword 8','Keyword 9','Keyword 10',
    #                                                         'Keyword 11','Keyword 12','Keyword 13','Keyword 14','Keyword 15'])

    # Convert the DataFrame to a string with a
    #
    # chart format
    best_keywords_chart = df_best_keywords.to_string(index=False)

    # Define the file path where the output will be saved
    output_file_path = 'best_keywords_chart.csv'

    # Write the string to a text file
    with open(output_file_path, 'w') as file:
        file.write(best_keywords_chart)

    # After the loop, write all the output data to the file at once
    txt_output_file_path = 'iteration_metrics.txt'
    with open(txt_output_file_path, 'w', encoding='utf-8') as txtfile:  # 'w' for write mode, overwriting the file
        txtfile.write('\n'.join(output_data))

    print("All metrics and sets have been written to the file.")
    print(f'The size of best keywords is {len(best_keywords)}')

    return best_performance, best_params, best_keywords

if __name__ == "__main__":
    # Call the function and unpack the results
    _, best_params, best_keywords = monte_carlo_with_parameter_sweeping()
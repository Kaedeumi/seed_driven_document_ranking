"""
    This file implements another way of retrieving the database which is using MLT functionality
    provided by ElasticSearch. Its results will be compared with the one obtained from Monte Carlo sampling.
"""

import json
from elasticsearch import Elasticsearch
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np

# Connect to your Elasticsearch cluster
# Replace with your actual username and password
username = 'yang'
password = 'theno1ofdmt'

# Connect to Elasticsearch with authentication details
es = Elasticsearch(
    hosts=["http://localhost:9200"],
    http_auth=(username, password)
)

def get_truly_pos_documents():
    '''
        This method finds the labeled positives in the training set and collects them as the truly_pos documents.
        These documents are later utilized for the purpose of retrieving the positive documents in the test set.

    :return: truly_pos_documents, which is a list of LP documents in the training set (which are the truly_poss)
    '''
    # List to store documents where 'label_L40' is "positive/labeled"
    truly_pos_documents = []
    file_path = '../data/pubmed-dse/L20/D000328.D008875.D015658/ordered_train.jsonl'
    # Open and read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Convert each line from JSON format to a Python dictionary
            document = json.loads(line)
            # Check if the 'label_L40' field equals "positive/labeled"
            if document.get('label_L40') == 'positive/labeled':
                # Add the document to the list if the condition is met
                truly_pos_documents.append(document)
    return truly_pos_documents

def get_test_set_positive_documents(test_path):
    '''
        This method finds the truly positives in the testing set.
        These documents are later utilized for the calculation of precision and recall.

    :return: truly_positive_docs, which is a list of truly positives in the testing set.
    '''
    # List to store documents where 'label_L40' is "positive/labeled"
    positive_documents = []
    file_path = test_path
    # Open and read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Convert each line from JSON format to a Python dictionary
            document = json.loads(line)
            # Check if the 'label_L40' field equals "positive/labeled"
            if document.get('label_true') == 'positive/labeled':
                # Add the document to the list if the condition is met
                positive_documents.append(document)
    return positive_documents

def get_test_set_labelled_positive_documents(test_path):
    '''
        This method finds the LABELLED positives in the testing set.
        These documents are later utilized for the calculation of precision and recall.

    :return: labeled_positive_docs, which is a list of labeled positives in the testing set.
    '''
    # List to store documents where 'label_L40' is "positive/labeled"
    positive_documents = []
    file_path = test_path
    # Open and read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Convert each line from JSON format to a Python dictionary
            document = json.loads(line)
            # Check if the 'label_L40' field equals "positive/labeled"
            if document.get('label_L40') == 'positive/labeled':
                # Add the document to the list if the condition is met
                positive_documents.append(document)
    return positive_documents

import random

def truly_pos_document_selection(L, truly_pos_corpus, truly_pos_value=42):
    """
    Randomly selects L documents from the truly_pos document corpus, ensuring the same selection every time by using a fixed truly_pos value.

    Parameters:
        L (int): The number of documents to select.
        truly_pos_corpus (list): The corpus of truly_pos documents to select from.
        truly_pos_value (int): The truly_pos value for the random number generator to ensure consistency.

    Returns:
        list: A list of consistently randomly selected documents.
    """
    # Check if L is greater than the number of documents in the truly_pos corpus
    if L > len(truly_pos_corpus):
        raise ValueError("L cannot be greater than the number of documents in the truly_pos corpus.")

    # Set the truly_pos for the random number generator
    random.seed(truly_pos_value)

    # Randomly select L documents from the truly_pos_corpus
    selected_documents = random.sample(truly_pos_corpus, L)

    # Reset the random truly_pos if necessary for other operations to remain truly random
    random.seed()  # This resets the random generator to its default state

    return selected_documents


def index_documents(jsonl_file_path, index_name):
    i = 0
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            i += 1
            doc = json.loads(line)
            res = es.index(index=index_name, document=doc)
            print('indexing document', i)
            print(res['result'])  # Prints whether the indexing operation was successful ('created')

# # Index documents (you only need to do this once)
testing_file_path = '../data/pubmed-dse/L20/D000328.D008875.D015658/ordered_test.jsonl'
# Replace 'your_index_name' with the actual name of your Elasticsearch index
index_name = "mlt_database4"

# index_documents(testing_file_path, index_name)
# Function to calculate precision and recall

# Assume 'results' contain the results from the MLT query
# Assume 'ground_truth_labels' contains the list of all relevant document IDs from your truly_pos set

def reading_test_jsonl_file(test_path):
    # Initialize an empty list to store the data
    data = []

    # Open and read the JSONL file
    with open(test_path, 'r') as file:
        for line in file:
            # Convert each line from JSON format to a Python dictionary
            entry = json.loads(line)
            # Add the 'order' and 'abstract' from each entry to the list
            data.append({
                'order': entry.get('order', 'NA'),  # Use 'NA' if 'order' is not found
                'abstract': entry.get('abstract', 'NA')  # Use 'NA' if 'abstract' is not found
            })

    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)

    # Write the DataFrame to a CSV file
    df.to_csv('test_docs.csv', index=False)

reading_test_jsonl_file(testing_file_path)
df_full = pd.read_csv('test_docs.csv')
# Calculate the number of rows for each percentage
num_rows = len(df_full)

def MLT_retrieval(index_name, output_csv_path):
    # Execute the MLT query
    try:
        results = es.search(index=index_name, body=mlt_query)
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    # Initialize a list to store rows for the DataFrame
    rows_list = []

    # Process the results
    for i, result in enumerate(results['hits']['hits'], start=1):
        label = result['_source'].get('label_true', 'unknown')
        order = result['_source'].get('order', 'unknown')
        score = result['_score']  # This is the BM25 score from Elasticsearch
        title = result['_source'].get('title', 'No title')
        abstract = result['_source'].get('abstract', 'null')

        # Print details including BM25 score for each document
        print(f"Doc No.{i} Label: {label}, Order: {order} BM25 Score: {score}, Title: {title}")

        # Append the row to the list
        rows_list.append({
            'Doc No.': f"Doc No.{i}",
            'Order' : order,
            'Label': label,
            'BM25 Score': score,  # Save the BM25 score in the DataFrame
            'Title': title,
            'Abstract': abstract
        })

    # Create a DataFrame from the rows list
    df_results = pd.DataFrame(rows_list)

    # Write the DataFrame to a CSV file
    df_results.to_csv(output_csv_path, index=False)

    print(f"Results have been written to {output_csv_path}")
    return results

def finding_pos_docs_and_rankings(retvd_pos_doc_list, retrieved_documents):
    '''
    This method finds the TP positive documents in the sorted document frequency list after a fixed parameter iteration.

    :param retvd_pos_doc_list: the list of orders that identify the truly_pos documents
    :param retrieved_documents: the dictionary containing retrieved documents by MLT searching

    :return: retvd_pos_doc_rankings: a table to be printed to a separate file which contains the truly positive documents found
    together with their rankings.
    '''
    # Find truly_pos documents in the retrieved documents and get their rankings
    retvd_pos_doc_rankings = []
    for retvd_pos_doc in retvd_pos_doc_list:
        # Count the number of retrieved documents that are relevant
        for rank, doc in enumerate(retrieved_documents['hits']['hits'][:retrieve_size],start=1):  # Consider the first 100 results
            if doc['_source'].get('order') == retvd_pos_doc:
                score = doc['_score']  # This is the BM25 score from Elasticsearch
                retvd_pos_doc_rankings.append({'doc_id': retvd_pos_doc, 'ranking': rank, 'BM25 score': score})
    return retvd_pos_doc_rankings

def finding_LP_docs_and_rankings(labeled_positive_doc_list, retrieved_documents):
    '''
    This method finds the Labeled Positive (LP) documents in the sorted document frequency list after a fixed parameter iteration.

    :param labeled_positive_doc_list: the list of orders that identify the LP documents
    :param retrieved_documents: the dictionary containing retrieved documents by MLT searching
    :return: LP_doc_rankings: a table to be printed to a separate file which contains the truly_pos documents found
    together with their rankings.
    '''
    # Find truly_pos documents in the retrieved documents and get their rankings
    LP_doc_rankings = []
    for LP_doc_id in labeled_positive_doc_list:
        # Count the number of retrieved documents that are relevant
        for rank, doc in enumerate(retrieved_documents['hits']['hits'][:retrieve_size],start=1):  # Consider the first 100 results
            if doc['_source'].get('order') == LP_doc_id:
                score = doc['_score']  # This is the BM25 score from Elasticsearch
                LP_doc_rankings.append({'doc_id': LP_doc_id, 'ranking': rank, 'BM25 score': score})
    return LP_doc_rankings

def calculate_truly_pos_performance_metrics(retrieved_documents, positive_documents):
    '''
    This method calculates the performance metrics, comparing the retrieved positive results with the truly positive results.

    :param retrieved_documents: a list of orders of the retrieved documents in the test set
    :param positive_documents: a list of orders of the truly positive documents in the test set
    :return: recall, precision, TP set
    '''

    retrieved_set = set(retrieved_documents)
    posdoc_set = set(positive_documents)
    true_positive_set = retrieved_set & posdoc_set

    # Calculate 'truly_pos recall' which is defined as TP/ the set of truly positive documents
    true_positives = len(set(retrieved_documents) & set(positive_documents))
    recall = true_positives / len(positive_documents) if positive_documents else 0

    # Calculate average truly_pos precision as TP/ the set of retrieved documents
    precision = true_positives / len(retrieved_documents) if retrieved_documents else 0

    return recall, precision, true_positive_set

def extracting_docids_from_results(results):
    # Extract the orders of the retrieved documents and construct them into a list
    results_order_list = []
    for doc in results['hits']['hits'][:retrieve_size]:
        res_order = doc['_source'].get('order')
        results_order_list.append(res_order)
    return results_order_list

def finding_truly_pos_docs_and_rankings_with_NaN(retvd_pos_doc_list, retrieved_documents):
    '''
    This method finds the TP positive documents in the sorted document frequency list after a fixed parameter iteration.

    :param retvd_pos_doc_list: the list of orders that identify the truly_pos documents
    :param retrieved_documents: the dictionary containing retrieved documents by MLT searching

    :return: retvd_pos_doc_rankings: a table to be printed to a separate file which contains the truly positive documents found
    together with their rankings.
    '''
    # Find truly_pos documents in the retrieved documents and get their rankings
    retvd_pos_doc_rankings = []

    # Create a dictionary from retrieved documents with 'order' as key and 'rank' as value
    retrieved_rankings = {doc['_source'].get('order'): rank
                          for rank, doc in enumerate(retrieved_documents['hits']['hits'], start=1)}

    # Go through each truly_pos document id
    for retvd_pos_doc in retvd_pos_doc_list:
        # Check if the truly_pos document is in the retrieved documents
        if retvd_pos_doc in retrieved_rankings:
            # If found, use the retrieved ranking
            rank = retrieved_rankings[retvd_pos_doc]
            mlt_score = retrieved_documents['hits']['hits'][rank - 1]['_score']
            retvd_pos_doc_rankings.append({'doc_id': retvd_pos_doc, 'ranking': rank, 'BM25 score': mlt_score})
        else:
            # If not found, assign NaN
            retvd_pos_doc_rankings.append({'doc_id': retvd_pos_doc, 'ranking': np.nan, 'BM25 score': np.nan})

    return retvd_pos_doc_rankings

def finding_LP_docs_and_rankings_with_NaN(labeled_positive_doc_list, retrieved_documents):
    """
    This method finds the truly_pos documents in the sorted document frequency list after a MLT querying

    :param labeled_positive_doc_list: the list of orders that identify the truly_pos documents
    :param retrieved_documents: the dictionary containing retrieved documents by MLT searching
    :return: truly_pos_doc_rankings: a table to be printed to a separate file which contains the truly_pos documents found
             together with their rankings.
    """
    # Find truly_pos documents in the retrieved documents and get their rankings
    LP_doc_rankings = []

    # Create a dictionary from retrieved documents with 'order' as key and 'rank' as value
    retrieved_rankings = {doc['_source'].get('order'): rank
                          for rank, doc in enumerate(retrieved_documents['hits']['hits'], start=1)}

    # Go through each truly_pos document id
    for LP_doc_id in labeled_positive_doc_list:
        # Check if the truly_pos document is in the retrieved documents
        if LP_doc_id in retrieved_rankings:
            # If found, use the retrieved ranking
            rank = retrieved_rankings[LP_doc_id]
            mlt_score = retrieved_documents['hits']['hits'][rank - 1]['_score']
            LP_doc_rankings.append({'doc_id': LP_doc_id, 'ranking': rank, 'BM25 score': mlt_score})
        else:
            # If not found, assign NaN
            LP_doc_rankings.append({'doc_id': LP_doc_id, 'ranking': np.nan, 'BM25 score': np.nan})

    return LP_doc_rankings

def list_exclusion(L1, L2):
    ''' This function removes all L1 elements in L2 to conduct exclusion for the convenience of plotting.'''
    # Check if L1 is a subset of L2
    if all(item in L2 for item in L1):
        # Create a copy of L2 to not modify the original list
        result = L2[:]
        # Remove elements of L1 from the copy of L2
        for item in L1:
            result.remove(item)
        return result
    else:
        # If L1 is not a subset of L2, return L2 unchanged
        return L2

def sensitivity_results_visualization(truly_pos_positions, L_values, percentage):
    '''
    This method visualizes the rankings of the truly_pos documents in the candidate retrieved list of documents
    executed by MLT search.
    :return:
    '''
    # Assuming you have the following data:
    # X-axis data - truly_pos corpus length (L)
    truly_pos_corpus_lengths = L_values

    # Y-axis data - positions in ranked list for truly_pos papers and relevant papers
    # These should be lists of lists, with each inner list containing positions for one L value
    truly_pos_paper_positions = truly_pos_positions
    # relevant_paper_positions = relevant_positions

    # Calculate averages and medians for relevant papers
    relevant_averages = [np.mean(positions) for positions in truly_pos_paper_positions]
    relevant_medians = [np.median(positions) for positions in truly_pos_paper_positions]

    # Start plotting
    plt.figure(figsize=(6,10))

    # Plot the truly_pos papers
    for i, length in enumerate(truly_pos_corpus_lengths):
        plt.scatter([length] * len(truly_pos_paper_positions[i]), truly_pos_paper_positions[i], color='blue',label='truly labeled positives' if i == 0 else "",zorder=2)

    # # Plot the relevant papers
    # for i, length in enumerate(truly_pos_corpus_lengths):
    #     plt.scatter([length] * len(relevant_paper_positions[i]), relevant_paper_positions[i], color='blue', label='LP Papers' if i == 0 else "", zorder=1)

    # Plot the relevant averages and medians
    plt.plot(truly_pos_corpus_lengths, relevant_averages, 'r--', label='Relevant Average', marker='D')
    plt.plot(truly_pos_corpus_lengths, relevant_medians, 'm--', label='Relevant Median', marker='^')

    # # Set the y-axis to log scale
    # plt.yscale('log')

    # Set the x-axis ticks and labels
    plt.xticks(truly_pos_corpus_lengths, truly_pos_corpus_lengths)

    # Label the axes
    plt.xlabel('truly_pos corpus length (L)')
    plt.ylabel('Position in Ranked List')

    # Add legend
    plt.legend()

    plt.savefig(f'MLT_document_rankings_{int(percentage * 100)}%.png')  # Save the plot as a .png file
    # Show the plot
    plt.show()

def iteration_performance_visualization(precision_dict, recall_dict):
    # Sort the keys to ensure the bars are in order
    parameter_combinations = sorted(precision_dict.keys())

    # Extract the precision and recall values in the same order
    precisions = [precision_dict[param] for param in parameter_combinations]
    recalls = [recall_dict[param] for param in parameter_combinations]


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
    rects1 = ax.bar(adjusted_x_precision, [p for p in precisions], width, label='Precision')

    # Recall bars remain unchanged
    rects2 = ax.bar(adjusted_x_recall, recalls, width, label='Recall')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('Parameter Combinations (percentage, L size)')
    ax.set_ylabel('Scores')
    ax.set_title('MLT: Precision and Recall by Parameter Combination')
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
    autolabel(rects1)

    # For recall bars, no scaling is applied so the function is called normally
    autolabel(rects2)

    # LaTeX annotations for Precision and Recall formulas
    formula_text = r'$\text{Precision} = \frac{n(\text{Retrieved} \cap \text{Truly Positive})}{n(\text{Retrieved})}$' + '\n' + \
                   r'$\text{Recall} = \frac{n(\text{Retrieved} \cap \text{Truly Positive})}{n(\text{Truly Positive})}$'
    ax.text(0.02, 0.98, formula_text, transform=ax.transAxes, fontsize=18, verticalalignment='top',
            bbox=dict(boxstyle="round", alpha=0.15))
    # Customize x-axis labels
    # plt.xticks(rotation=30, ha='right')  # Rotate the labels and align them to the right
    plt.tight_layout()  # Adjust the padding between and around subplots.
    # Other plotting code remains unchanged
    # Show the plot
    plt.savefig('MLT_parameter_sweeping.png')  # Save the plot as a .png file
    plt.show()


# Assume 'results' contain the results from the MLT query
# Assume 'ground_truth_labels' contains the list of all relevant document IDs from your truly_pos set
# Load the labeled positives

percentages  = [0.1,0.3, 0.5]
precision_dict = {}
recall_dict = {}

for percentage in percentages:
    retrieve_size = int(num_rows * percentage)

    # get the truly_pos document set from the training set
    truly_pos_corpus = get_truly_pos_documents()

    # Random selection of L instances of truly_pos documents
    # Set the size L, the number of random instances to be picked

    L_list = [1,8,20]
    pos_positions_allIters = [] # it's assumed to be a list of lists
    LP_positions_allIters = [] # also a list of lists

    for L_value in L_list:
        '''
            This loop iterates through different L sizes to investigate the effect of the ranking list caused by L lengths.
            the truly_pos_documents variable represents the selected portion of the truly_pos corpus.
        '''

        truly_pos_documents = truly_pos_document_selection(L_value, truly_pos_corpus)
        # print(f'truly_pos documents found for L = {L_value} is {truly_pos_documents}')
        # for truly_pos in truly_pos_documents:
        #     print(truly_pos)
        # print('\n')

        # Before constructing the query, make sure to remove the 'Order' field if it's there
        # For example, if truly_pos_documents is a list of dictionaries, do:
        mlt_truly_pos_texts = [doc['abstract'] for doc in truly_pos_documents]
        truly_pos_text_orders = [doc['order'] for doc in truly_pos_documents]
        print(f'the order of the truly_pos texts are {truly_pos_text_orders}')

        # full_truly_pos_texts = [doc['Abstract'] for doc in labeled_positive]
        # labeled_positive_orders = [doc['Identifier'] for doc in labeled_positive]
        # print(f'The orders of the labeled positive documents are \n {labeled_positive_orders}')

        # Now, construct the MLT query using the truly_pos documents
        mlt_query = {
            "query": {
                "more_like_this": {
                    "fields": ["abstract"],
                    "like": mlt_truly_pos_texts,
                    "min_term_freq": 1,
                    "min_doc_freq": 1,
                }
            },
            "size": retrieve_size  # You can change the size to retrieve more or fewer documents
        }


        MLTresult = f'MLTresults/percentage = {percentage}, L = {L_value}.csv'
        results = MLT_retrieval(index_name,MLTresult)  # Call your function to retrieve MLT results

        results_order_list = extracting_docids_from_results(results)
        # print(results_order_list)

        # extract the truly positive text orders out for performance evaluation
        truly_positives = get_test_set_positive_documents(testing_file_path)
        truly_positive_text_orders = [doc['order'] for doc in truly_positives]
        truly_positive_text_orders.sort()
        print(f'The truly positive set contains the documents {truly_positive_text_orders}')
        truly_pos_recall, truly_pos_precision, TPs = calculate_truly_pos_performance_metrics(results_order_list,truly_positive_text_orders)

        # Print precision and recall
        print(f'Iteration of seed size L = {L_value}')
        print('The precision and recall performance with regard to seed documents is =======================')
        print(f'truly_pos Precision is {truly_pos_precision}, truly_pos Recall is {truly_pos_recall},\n the true positive set (doc id) is {TPs}')

        precision_dict[(percentage, L_value)] = truly_pos_precision
        recall_dict[(percentage, L_value)] = truly_pos_recall

        iteration_performance_visualization(precision_dict, recall_dict)

        pos_doc_rankings = finding_pos_docs_and_rankings(truly_positive_text_orders,results)
        # Convert the rankings to a DataFrame and save to a CSV file
        rankings_df = pd.DataFrame(pos_doc_rankings)
        rankings_df.to_csv(f'L_iterations/retvd_pos_document_rankings_perc={percentage},L={L_value}.csv', index=False)

        truly_pos_doc_rankings2 = finding_truly_pos_docs_and_rankings_with_NaN(truly_positive_text_orders,results)
        # Convert the rankings to a DataFrame and save to a CSV file
        rankings_df_with_nan = pd.DataFrame(truly_pos_doc_rankings2)
        rankings_df_with_nan.to_csv(f'L_iterations/retvd_pos_document_rankings_extended_L={L_value}.csv', index=False)

        # Extract the 'ranking' values into a list
        pos_positions = [doc['ranking'] for doc in pos_doc_rankings]
        print(pos_positions)

        # find the (manually) labeled positives in the test set
        labeled_positives = get_test_set_labelled_positive_documents(testing_file_path)
        labeled_positive_orders = [doc['order'] for doc in labeled_positives]
        print(f'labeled positive orders in test set are \n {labeled_positive_orders}')


        LP_document_rankings = finding_LP_docs_and_rankings(labeled_positive_orders,results)
        # print(LP_document_rankings)
        LP_rankings_df = pd.DataFrame(LP_document_rankings)
        LP_rankings_df.to_csv(f'L_iterations/LP_document_rankings_perc={percentage}_L={L_value}.csv', index=False)

        LP_positions = [doc['ranking'] for doc in LP_document_rankings]
        print(LP_positions)

        LP_document_rankings2 = finding_LP_docs_and_rankings_with_NaN(labeled_positive_orders,results)
        LP_rankings_df_with_nan = pd.DataFrame(LP_document_rankings2)
        LP_rankings_df_with_nan.to_csv(f'L_iterations/LP_document_rankings_extended_L={L_value}.csv', index=False)

        pos_positions_allIters.append(pos_positions)
#     LP_positions_allIters.append(LP_positions)
#
# # # Plot the graph
    sensitivity_results_visualization(pos_positions_allIters, L_list,percentage)
#
# # Evaluation with BM25==================================================================================
# import pandas as pd
# from rank_bm25 import BM25Okapi
# from elasticsearch import Elasticsearch
#
#
# # Function to convert search results to a DataFrame
# def results_to_dataframe(results):
#     rows_list = []
#     for i, result in enumerate(results['hits']['hits'], start=1):
#         rows_list.append({
#             'Doc No.': f"Doc No.{i}",
#             'Label': result['_source'].get('label_true', 'unknown'),
#             'Order': result['_source'].get('order', 'unknown'),
#             'Score': result['_score'],
#             'Title': result['_source'].get('title', 'No title'),
#             'Text': result['_source'].get('abstract', '')  # or whatever field contains the document text
#         })
#     return pd.DataFrame(rows_list)
#
#
# # Function to evaluate relevance with BM25
# def evaluate_bm25(truly_pos_text_list, documents_dataframe):
#     # Tokenize the truly_pos documents
#     tokenized_truly_pos_texts = [doc.split(" ") for doc in truly_pos_text_list]
#
#     # Tokenize the documents
#     tokenized_docs = [doc.split(" ") for doc in documents_dataframe['Text'].tolist()]
#
#     # Create a BM25 object
#     bm25 = BM25Okapi(tokenized_docs)
#
#     # Calculate scores for each truly_pos text
#     bm25_scores = [bm25.get_scores(text) for text in tokenized_truly_pos_texts]
#
#     # Average the scores for a simple relevance metric
#     # This is a simplification, more sophisticated methods could be used
#     avg_scores = [sum(scores) / len(scores) for scores in bm25_scores]
#
#     return avg_scores
#
#
# # Assuming 'es' is an instance of Elasticsearch
# # Assuming 'mlt_query' is your MLT query
# # Assuming 'index_name' is the name of your index
#
# # Retrieve results
# results = es.search(index=index_name, body=mlt_query)
#
# # Convert results to a DataFrame
# documents_dataframe = results_to_dataframe(results)
#
# # Assuming 'positive_texts' contains the text of the truly_pos documents
# truly_pos_text_list = mlt_truly_pos_texts  # Replace with actual list if different
#
# # Evaluate relevance with BM25
# bm25_relevance_scores = evaluate_bm25(truly_pos_text_list, documents_dataframe)
#
# # Sort the documents by BM25 relevance scores in descending order
# sorted_doc_scores = sorted(enumerate(bm25_relevance_scores), key=lambda x: x[1], reverse=True)
#
# # Limit to the top 100 documents
# top_100_doc_scores = sorted_doc_scores[:100]
#
# # Print BM25 relevance scores for the top 100 documents
# print("Top 100 Docs with Highest Relevance Scores:")
# for rank, (doc_id, score) in enumerate(top_100_doc_scores, start=1):
#     print(f"Rank {rank} Doc No.{doc_id + 1} BM25 Relevance Score: {score}")

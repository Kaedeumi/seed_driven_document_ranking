import pandas as pd
import math
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from MCMC import vectorizer, truly_positive_docs, test_path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json


# Function to preprocess text: tokenize, remove stopwords, stem/lemmatize
lemmatizer = WordNetLemmatizer()
truly_positive_text_orders = [doc['order'] for doc in truly_positive_docs]

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

def calculate_truly_pos_performance_metrics(retrieved_documents, truly_pos_documents):
    '''
    In this method, it's implicitly assumed that the retrieved_documents and truly_pos_documents are both list of strings.

    :param retrieved_documents: list of strings that represent the documents retrieved after each query.
    ** In this occasion, it's implicitly assumed that these documents are the ones on the ranked candidate list.""
    :param truly_pos_documents: list of strings that represent the set of truly positive documents defined.
    :return: truly_pos recall and precision for each query
    '''

    retrieved_set = set(retrieved_documents)
    truly_pos_set = set(truly_pos_documents)
    true_positive_set = retrieved_set & truly_pos_set

    # Calculate truly_pos recall
    true_positives = len(set(retrieved_documents) & set(truly_pos_documents))
    recall = true_positives / len(truly_pos_documents) if truly_pos_documents else 0

    # Calculate average truly_pos precision
    precision = true_positives / len(retrieved_documents) if retrieved_documents else 0

    return recall, precision, true_positive_set

def get_bm25_score(document_order, df):
    """
    Get the BM25 score for a given document order.

    :param document_order: The order of the document for which to find the BM25 score.
    :param df: A pandas DataFrame containing document orders and their BM25 scores.
    :return: The BM25 score for the given document order or None if not found.
    """
    # Check if the document_order exists in the DataFrame
    result = df[df['Document Order'] == document_order]

    # If the document order is found, return the BM25 score, otherwise return None
    return result['bm25_score'].iloc[0] if not result.empty else None

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

def finding_truly_pos_docs_and_rankings(truly_pos_doc_list, retrieved_documents):
    """
    Find the truly_pos documents in the BM25 ranked list and get their rankings.

    :param truly_pos_doc_list: A list of document orders that identify the truly_pos documents.
    :param retrieved_documents: A list of orders of the retrieved documents in the candidate list.
    :return: A table with the truly_pos documents and their BM25 rankings.
    """
    # Load the BM25 scores from the CSV file
    df_bm25 = pd.read_csv('bm25_scores.csv')

    # Sort the DataFrame by BM25 score in descending order
    df_bm25_sorted = df_bm25.sort_values('bm25_score', ascending=False).reset_index(drop=True)

    # Create a dictionary mapping from document order to BM25 ranking
    order_to_ranking = pd.Series(df_bm25_sorted.index + 1, index=df_bm25_sorted['Document Order']).to_dict()

    # Find the rankings of the truly_pos documents based on the sorted BM25 scores
    truly_pos_doc_rankings = []
    for truly_pos_doc_id in truly_pos_doc_list:
        rank = order_to_ranking.get(truly_pos_doc_id, float('nan'))  # Default to NaN if not found
        score = df_bm25_sorted.loc[df_bm25_sorted['Document Order'] == truly_pos_doc_id, 'bm25_score'].iloc[
            0] if truly_pos_doc_id in order_to_ranking else float('nan')
        truly_pos_doc_rankings.append({'Document Order': truly_pos_doc_id, 'ranking': rank, 'bm25_score': score})

    return truly_pos_doc_rankings


def finding_truly_pos_docs_and_rankings_without_nan(truly_pos_doc_list, nkw):
    """
    Find the truly_pos documents in the BM25 ranked list and get their rankings.
    Only includes documents that are found in the rankings.

    :param truly_pos_doc_list: A list of document orders that identify the truly_pos documents.
    :param retrieved_documents: A list of orders of the retrieved documents in the candidate list.
    :return: A table with the truly_pos documents and their BM25 rankings if found.
    """
    # Load the BM25 scores from the CSV file
    df_bm25 = pd.read_csv(f'bm25_scores_nkw{nkw}.csv')

    # Sort the DataFrame by BM25 score in descending order
    df_bm25_sorted = df_bm25.sort_values(by='bm25_score', ascending=False).reset_index(drop=True)

    # Create a dictionary mapping from document order to BM25 ranking
    order_to_ranking = pd.Series(df_bm25_sorted.index + 1, index=df_bm25_sorted['Document Order']).to_dict()

    # Find the rankings of the truly_pos documents based on the sorted BM25 scores
    truly_pos_doc_rankings = []
    for truly_pos_doc_id in truly_pos_doc_list:
        if truly_pos_doc_id in order_to_ranking:
            rank = order_to_ranking[truly_pos_doc_id]
            score = df_bm25_sorted.loc[df_bm25_sorted['Document Order'] == truly_pos_doc_id, 'bm25_score'].iloc[0]
            truly_pos_doc_rankings.append({'Document Order': truly_pos_doc_id, 'ranking': rank, 'bm25_score': score})

    return truly_pos_doc_rankings


def finding_LP_docs_and_rankings(labeled_positive_doc_list, nkw):
    """
     Find the truly_pos documents in the BM25 ranked list and get their rankings.

     :param labeled_positive_doc_list: A list of document orders that identify the LP documents.
     :param retrieved_documents: A list of orders of the retrieved documents in the candidate list.
     :return: A table with the truly_pos documents and their BM25 rankings.
     """
    # Load the BM25 scores from the CSV file
    df_bm25 = pd.read_csv(f'bm25_scores_nkw{nkw}.csv')

    # Sort the DataFrame by BM25 score in descending order
    df_bm25_sorted = df_bm25.sort_values('bm25_score', ascending=False).reset_index(drop=True)

    # Create a dictionary mapping from document order to BM25 ranking
    order_to_ranking = pd.Series(df_bm25_sorted.index + 1, index=df_bm25_sorted['Document Order']).to_dict()

    # Find the rankings of the truly_pos documents based on the sorted BM25 scores
    lp_doc_rankings = []
    for lp_doc_id in labeled_positive_doc_list:
        rank = order_to_ranking.get(lp_doc_id, float('nan'))  # Default to NaN if not found
        score = df_bm25_sorted.loc[df_bm25_sorted['Document Order'] == lp_doc_id, 'bm25_score'].iloc[
            0] if lp_doc_id in order_to_ranking else float('nan')
        lp_doc_rankings.append({'Document Order': lp_doc_id, 'ranking': rank, 'bm25_score': score})

    return lp_doc_rankings

def finding_lp_docs_and_rankings_without_nan(lp_doc_list, nkw):
    """
    Find the truly_pos documents in the BM25 ranked list and get their rankings.
    Only includes documents that are found in the rankings.

    :param truly_pos_doc_list: A list of document orders that identify the truly_pos documents.
    :param retrieved_documents: A list of orders of the retrieved documents in the candidate list.
    :return: A table with the truly_pos documents and their BM25 rankings if found.
    """
    # Load the BM25 scores from the CSV file
    df_bm25 = pd.read_csv(f'bm25_scores_nkw{nkw}.csv')

    # Sort the DataFrame by BM25 score in descending order
    df_bm25_sorted = df_bm25.sort_values(by='bm25_score', ascending=False).reset_index(drop=True)

    # Create a dictionary mapping from document order to BM25 ranking
    order_to_ranking = pd.Series(df_bm25_sorted.index + 1, index=df_bm25_sorted['Document Order']).to_dict()

    # Find the rankings of the truly_pos documents based on the sorted BM25 scores
    lp_doc_rankings = []
    for lp_doc_id in lp_doc_list:
        if lp_doc_id in order_to_ranking:
            rank = order_to_ranking[lp_doc_id]
            score = df_bm25_sorted.loc[df_bm25_sorted['Document Order'] == lp_doc_id, 'bm25_score'].iloc[0]
            lp_doc_rankings.append({'Document Order': lp_doc_id, 'ranking': rank, 'bm25_score': score})

    return lp_doc_rankings
def sensitivity_results_visualization(truly_pos_positions, n_kw, relevant_positions):
    '''
    This method visualizes the rankings of the truly_pos documents in the candidate retrieved list of documents
    executed by MLT search.
    :return:
    '''
    # Assuming you have the following data:
    # X-axis data - truly_pos corpus length (L)
    truly_pos_corpus_lengths = n_kw

    # Y-axis data - positions in ranked list for truly_pos papers and relevant papers
    # These should be lists of lists, with each inner list containing positions for one L value
    truly_pos_paper_positions = truly_pos_positions
    relevant_paper_positions = relevant_positions

    # Calculate averages and medians for relevant papers
    relevant_averages = [np.mean(positions) for positions in truly_pos_paper_positions]
    relevant_medians = [np.median(positions) for positions in truly_pos_paper_positions]

    # for relevant_papers in relevant_paper_positions:
    #     print(relevant_papers)

    print(f'the relevant averages are {relevant_averages}')
    print(f'the relevant medians are {relevant_medians}')

    # Start plotting
    plt.figure(figsize=(6,10))

    # Plot the truly_pos papers
    for i, length in enumerate(truly_pos_corpus_lengths):
        plt.scatter([length] * len(truly_pos_paper_positions[i]), truly_pos_paper_positions[i], color='orange', alpha=0.1, label='truly_pos Papers' if i == 0 else "", zorder=1)

    # Plot the relevant papers
    for i, length in enumerate(truly_pos_corpus_lengths):
        plt.scatter([length] * len(relevant_paper_positions[i]), relevant_paper_positions[i], color='blue', alpha=1,label='LP Papers' if i == 0 else "", zorder=2)

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
    plt.savefig('MC_document_rankings.png')  # Save the plot as a .png file
    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

def doc_position_distribution_plot(bm25_scores, df_scores, truly_pos_found_positions, relevant_found_positions, normalized_relevance):
    '''
    # Assuming 'bm25_scores' and 'df_scores' are lists or arrays with the BM25 and df values
    # 'truly_pos_found_positions' and 'relevant_found_positions' are the positions where truly_pos and relevant documents are found
    # 'normalized_relevance' is a list or array with the normalized relevance values for each document

    :param bm25_scores:
    :param df_scores:
    :param truly_pos_found_positions:
    :param relevant_found_positions:
    :param normalized_relevance:
    :return:
    '''
    # Create a figure and axis
    plt.figure(figsize=(10, 5))

    # Line plot for BM25 and document frequency (df) across the ranks of documents
    plt.plot(range(len(bm25_scores)), bm25_scores, label='bm25', color='blue')
    plt.plot(range(len(df_scores)), df_scores, label='df', color='purple')

    # Scatter plot for the positions where 'truly_pos Found' and 'Relevant Found' documents are located
    plt.scatter(truly_pos_found_positions, [normalized_relevance[pos] for pos in truly_pos_found_positions], color='orange',
                label='truly_pos FOUND', marker='D')
    plt.scatter(relevant_found_positions, [normalized_relevance[pos] for pos in relevant_found_positions],
                color='green', label='Relevant FOUND', marker='s')

    # Set the x and y-axis labels
    plt.xlabel('Document Rank in Ordered List')
    plt.ylabel('Normalized Relevance')

    # Add a legend
    plt.legend()

    # Optionally, set the x-axis to a logarithmic scale
    # plt.xscale('log')

    # Show the plot
    plt.show()

def bm25_score_calculation(nkw_list):
    '''
    This method calculates the BM25 scores of each document in the candidate list with respect to the
    feature vector Q.

    :param nkw_list: the range of length of keywords
    :return: list of lists of truly_pos and LP positions
    '''
    truly_pos_positions_allIters = []
    lp_positions_allIters = []

    for nkw in nkw_list:
        # Read the CSV file
        df = pd.read_csv(f'parameter_sweeping/document_frequencies_nmc1000_nkw{nkw}.csv')

        # Read the orders of the retrieved documents
        # Load the CSV file into a DataFrame

        # Extract the 'Document Order' column and convert it to a list
        document_order_list = df['Document Order'].tolist()

        # Now document_order_list contains all the 'Document Order' values from the CSV
        print(document_order_list)

        print(truly_positive_text_orders)

        recall, precision, TP = calculate_truly_pos_performance_metrics(document_order_list, truly_positive_text_orders)
        print(f'The truly_pos recall in the candidate list is {recall}')
        print(f'The true positive set has a size of {len(TP)}, \n containing {TP}')

        # Assuming feature vector Q is already defined and preprocessed
        feature_vector_Q = vectorizer  # This should be a list of preprocessed terms

        # Preprocess abstracts from the CSV file
        df['processed_abstract'] = df['Abstract'].apply(preprocess_text)

        # Calculate IDF for each term in the feature vector across all abstracts
        documents = df['processed_abstract'].tolist()
        # print(documents)

        avg_doc_length = sum(len(doc) for doc in documents) / len(documents)
        # Accessing feature names from the TF-IDF matrix
        feature_names = vectorizer.get_feature_names_out()

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
        df.to_csv(f'bm25_scores_nkw{nkw}.csv', index=False)

        truly_pos_doc_rankings = finding_truly_pos_docs_and_rankings_without_nan(truly_positive_text_orders, nkw)
        # Convert the rankings to a DataFrame and save to a CSV file
        rankings_df = pd.DataFrame(truly_pos_doc_rankings)
        rankings_df.to_csv(f'candidate_list_evaluation/truly_pos_document_rankings_nkw{nkw}.csv', index=False)
        # Extract the 'ranking' values into a list
        truly_pos_positions = [doc['ranking'] for doc in truly_pos_doc_rankings]
        print(truly_pos_positions)

        # find the LABELED POSITIVE documents in the test set
        LP_documents = get_test_set_labelled_positive_documents(test_path)
        # Extract the orders from the LP documents
        labeled_positive_text_orders = [doc['order'] for doc in LP_documents]

        lp_doc_rankings = finding_lp_docs_and_rankings_without_nan(labeled_positive_text_orders, nkw)
        lp_rankings = pd.DataFrame(lp_doc_rankings)
        lp_rankings.to_csv(f'candidate_list_evaluation/lp_document_rankings_nkw{nkw}.csv', index=False)
        LP_positions = [doc['ranking'] for doc in lp_doc_rankings]
        print(LP_positions)
        #
        # Add a column representing the normalized scores based on the files above
        # Load the data into a DataFrame
        df_append = pd.read_csv(f'bm25_scores_nkw{nkw}.csv')

        # Calculate the normalized BM25 score
        df_append['normalized_bm25_score'] = (df_append['bm25_score'] - df_append['bm25_score'].min()) / (
                df_append['bm25_score'].max() - df_append['bm25_score'].min())

        # Save the updated DataFrame back to a CSV
        df_append.to_csv(f'BM25_scores_with_norm/lp_document_rankings_nkw{nkw}_normed.csv', index=False)

        truly_pos_positions_allIters.append(truly_pos_positions)
        lp_positions_allIters.append(LP_positions)

    return truly_pos_positions_allIters, lp_positions_allIters


nkw_list = [5, 10, 15]
bm25_score_calculation(nkw_list)

truly_pos_positions_allIters, lp_positions_allIters = bm25_score_calculation(nkw_list)
truly_pos_positions = truly_pos_positions_allIters[1]
lp_positions = lp_positions_allIters[1]

normed_df_path = 'BM25_scores_with_norm/lp_document_rankings_nkw10_normed.csv'
# Load the CSV file into a DataFrame
df = pd.read_csv(normed_df_path)
# Extract the column and convert it to a list
df_scores = df['Raw DF'].tolist()
normed_df = df['Normalized Frequency'].tolist()

bm25_scores = df['bm25_score'].tolist()
normed_bm25 = df['normalized_bm25_score'].tolist()

if __name__ == "__main__":
    sensitivity_results_visualization(truly_pos_positions_allIters, nkw_list, lp_positions_allIters)

    # Now we only analyze the relative position distributions of truly_pos documents and LP documents when evaluated by df and BM25 metrics
    # with a fixed parameter value nkw = 10.

    # doc_position_distribution_plot(bm25_scores, df_scores, truly_pos_positions, lp_positions, normed_bm25)

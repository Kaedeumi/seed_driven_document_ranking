'''
    This code script analyses the positions of truly positive documents of different L sizes given a pair of the
    best performance parameter n_mc and n_kw.
'''

import json
import MCMC
from MCMC import test_path, truly_positive_docs
import pandas as pd
import re
import numpy as np

# # Now you can use best_params as you need
# print(keywordExtraction_and_MC_sampling.best_params)
# best_params = keywordExtraction_and_MC_sampling.best_params
# print(best_params)
# best_n_mc = best_params['N_MC']
# print(best_n_mc)
# best_n_kw = best_params['N_kw']
# print(best_n_kw)

# # for simplicity, assume that
best_n_mc = 1000
best_n_kw = 5

def extracting_orders_from_truly_pos_docs(truly_positive_docs):
    # Extract the 'order' entries from the truly labeled positive documents
    truly_positive_text_orders = [doc['order'] for doc in truly_positive_docs]
    return truly_positive_text_orders

def finding_truly_positive_docs_and_rankings(best_n_mc, best_n_kw):
    '''
    This method finds the truly positive documents in the sorted document frequency list after a fixed parameter iteration.

    :param best_n_mc
    :param best_n_kw
    :return: the file path from which the data is acquired
    '''

    # The path to the directory containing the CSV files (assuming the current directory)
    directory_path = 'parameter_sweeping/'

    # Construct the file name based on the best parameters
    file_name = f'document_frequencies_nmc{best_n_mc}_nkw{best_n_kw}.csv'
    file_path = directory_path + file_name

    # Read the CSV file
    df = pd.read_csv(file_path)

    # List of seed document identifiers
    truly_pos_document_identifiers = extracting_orders_from_truly_pos_docs(truly_positive_docs)

    # Find the rankings of the seed documents
    truly_pos_document_rankings = {identifier: df[df['Document Order'] == identifier].index[0] + 1
                              for identifier in truly_pos_document_identifiers
                              if identifier in df['Document Order'].values}

    # Assuming seed_document_rankings is a dictionary with identifiers as keys and rankings as values
    # total_iterations is the length of the dictionary

    total_iterations = len(truly_pos_document_rankings)

    # Output the rankings
    for i, (identifier, ranking) in enumerate(truly_pos_document_rankings.items(), start=1):
        print(f'Iteration {i}/{total_iterations}: The ranking of truly positive document {identifier} is {ranking}')

    return file_path,truly_pos_document_rankings

finding_truly_positive_docs_and_rankings(best_n_mc,best_n_kw)

def ranking_of_truly_pos_corpus_in_candidates_multi(best_n_mc, best_n_kw):
    '''
    This method generates a chart summarizing the positions of the truly positive documents in the retrieved results,
    which looks like the one in the supplementary material Table S4, but with only single column of N_kw.

    :return:
    '''
    # Get the labeled positive document identifiers
    identifiers = extracting_orders_from_truly_pos_docs(truly_positive_docs)

    # Get the path to the CSV file
    file_path, _ = finding_truly_positive_docs_and_rankings(best_n_mc, best_n_kw)

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Create a DataFrame to compile the rankings with a column for identifiers
    rankings_df = pd.DataFrame(columns=['Identifier', f'N_kw = {best_n_kw}'])

    # Compile the rankings
    for identifier in identifiers:
        if identifier in df['Document Order'].values:
            # Get the ranking of the identifier and cast to int
            ranking = int(df.index[df['Document Order'] == identifier][0] + 1)
        else:
            # If the identifier is not found, set it as NaN
            ranking = np.nan  # Use "#N/A" if you prefer to represent missing values as such

        # Add the identifier and ranking to the compiled DataFrame
        rankings_df = rankings_df._append({'Identifier': identifier, f'N_kw = {best_n_kw}': ranking}, ignore_index=True)

    # Replace NaN with "#N/A" if needed
    rankings_df.fillna('#N/A', inplace=True)

    # Define the try_convert_to_int function to handle conversion and exceptions
    def try_convert_to_int(x):
        try:
            return int(x)  # Try to convert to int
        except ValueError:
            return x  # If a ValueError occurs, return the original value

    # Use a try-except block within the lambda function to handle the conversion and the potential ValueError
    rankings_df[f'N_kw = {best_n_kw}'] = rankings_df[f'N_kw = {best_n_kw}'].apply(lambda x: try_convert_to_int(x))

    def format_ranking(value):
        try:
            # If value can be converted to a float then to an int, it's numeric
            return int(float(value))
        except (ValueError, TypeError):
            # If it can't be converted to float, return it as it is (e.g., for '#N/A')
            return value

    # Apply the formatting to each column except 'Identifier'
    for col in rankings_df.columns:
        if col != 'Identifier':
            rankings_df[col] = rankings_df[col].apply(format_ranking)

    # Output the compiled DataFrame to a CSV file
    output_csv_path = f'truly_pos_docs_ranking/compiled_rankings_nmc{best_n_mc}_mkw{best_n_kw}.csv'
    rankings_df.to_csv(output_csv_path, index=False, na_rep='#N/A')

    print(f'Compiled rankings chart saved to {output_csv_path}')


def ranking_of_truly_pos_corpus_in_candidates_all_in_one(best_n_mc, best_n_kws):
    truly_pos_document_identifiers = extracting_orders_from_truly_pos_docs(truly_positive_docs)
    rankings_df = pd.DataFrame(index=truly_pos_document_identifiers)

    for n_kw in best_n_kws:
        file_path = f'truly_pos_docs_ranking/document_frequencies_nmc{best_n_mc}_nkw{n_kw}.csv'  # Adjust path as needed
        _, rankings_df[f'N_kw = {n_kw}'] = finding_truly_positive_docs_and_rankings(best_n_mc,n_kw)

    # Replace NaN values with '#N/A'
    rankings_df = rankings_df.fillna('#N/A')

    def format_ranking(value):
        try:
            # If value can be converted to a float then to an int, it's numeric
            return int(float(value))
        except (ValueError, TypeError):
            # If it can't be converted to float, return it as it is (e.g., for '#N/A')
            return value

    # Apply the formatting to each column except 'Identifier'
    for col in rankings_df.columns:
        if col != 'Identifier':
            rankings_df[col] = rankings_df[col].apply(format_ranking)


    # Save the DataFrame to a CSV file
    output_csv_path = 'compiled_rankings.csv'
    rankings_df.to_csv(output_csv_path, index_label='Identifier')

    return output_csv_path

# Example usage:
#
best_n_kws = [5,10,15]  # Example values, replace with actual
ranking_of_truly_pos_corpus_in_candidates_all_in_one(best_n_mc, best_n_kws)


for n in (5,10,15):
    ranking_of_truly_pos_corpus_in_candidates_multi(best_n_mc, n)

# keywordExtraction_and_MC_sampling.monte_carlo_sampling_with_fixed_params(600,5)
# keywordExtraction_and_MC_sampling.monte_carlo_sampling_with_fixed_params(600,15)
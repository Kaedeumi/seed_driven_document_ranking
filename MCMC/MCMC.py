import json
import random
from collections import OrderedDict

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import csv
import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from itertools import product

from Monte_carlo_sampling.keywordExtraction_and_MC_sampling import lemmatizer

# Connect to your Elasticsearch cluster
# Replace with your actual username and password
username = 'yang'
password = 'theno1ofdmt'

# Connect to Elasticsearch with authentication details
es = Elasticsearch(
    hosts=["http://localhost:9200"],
    http_auth=(username, password)
)

df = pd.read_csv('../Monte_carlo_sampling_new/terms_with_probabilities.csv')
index_name = "mlt_database4"

# Function to perform one step of Metropolis-Hastings for categorical data
def metropolis_hastings_step(current_state_index, df, counter, writer):
    # Propose a new state (a new keyword) uniformly at random
    proposed_state_index = np.random.choice(df.index)

    # Compute acceptance probability
    current_probability = df.iloc[current_state_index]['Probability']
    current_term = df.iloc[current_state_index]['Term']
    proposed_probability = df.iloc[proposed_state_index]['Probability']
    acceptance_probability = min(1, proposed_probability / current_probability)

    accepted = np.random.rand() < acceptance_probability

    print(f'The no. {counter} randomly chosen state is {proposed_state_index}, current prob is {current_probability}.'
          f'proposed prob is {proposed_probability}, sample accepted? {accepted}')

    writer.writerow([counter, proposed_state_index, current_probability, proposed_probability, accepted])

    # Accept or reject the new state
    if accepted:
        print(f'accepted: {current_term}')
        return proposed_state_index  # Accept the new state
    else:
        return current_state_index  # Reject the new state and stay in current state

def send_query_to_database(keywords, n_kw):
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

# Function to run the MCMC sampler
def run_mcmc_sampler(df, num_iterations, start_state_index=0):
    # Open a CSV file to write the results
    with open('mcmc_output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header of the CSV file
        writer.writerow(['Iteration', 'State', 'Current Probability', 'Proposed Probability', 'Accepted'])

        state_index = start_state_index
        samples = []
        counter = 0
        converged_samples = []
        for _ in range(num_iterations):
            counter += 1
            state_index = metropolis_hastings_step(state_index, df, counter, writer)

            if _ > 200:
                if _ < num_iterations - 25:
                    # discarding burn-in states
                    samples.append(df.iloc[state_index]['Term'])
                else:
                    samples.append(df.iloc[state_index]['Term'])
                    converged_samples.append(df.iloc[state_index]['Term'])

        return samples, converged_samples


def output_samples_to_csv(samples, output_file):
    # Count the occurrences of each word in samples
    word_counts = {}
    for word in samples:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

    # Write the word counts to a CSV file
    with open(output_file, 'w', newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Term', 'Count'])  # Header
        for word, count in word_counts.items():
            writer.writerow([word, count])

def distribution_plot(df1, df2, color1, color2, iterations):
    # Plotting the probability distribution across terms
    plt.figure(figsize=(14, 7))  # You can adjust the figure size as needed
    plt.bar(df1['order'], df1['Probability'], color=color1, width=10, label='Original Distribution')
    plt.bar(df2['order'], df2['Probability'], color=color2, width=10, alpha=0.5, label='MCMC Distribution')
    plt.xlabel('Terms', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    # plt.xticks(rotation=90)  # Rotate the x labels for better readability
    plt.title(f'Probability Distribution Across Terms : Iteration = {iterations}', fontsize=14)
    plt.tight_layout()  # Adjust layout to ensure labels and titles are readable
    plt.legend()  # Show legend
    plt.savefig('combined_probability_distribution.png')  # Save the plot as a .png file
    plt.show()

def monte_carlo_sampling(sorted_terms, probabilities, N_kw):
    sampled_keywords = random.choices(sorted_terms, weights=probabilities, k=N_kw)
    return sampled_keywords

# Run the sampler
num_iterations = 500000
start_state_index = 0  # Starting from the first keyword in the dataframe
samples, converged_samples = run_mcmc_sampler(df, num_iterations, start_state_index)

# print(f'The samples are {samples}')
output_file_path = 'MCMC_results/mcmc_iterations.csv'
output_samples_to_csv(samples, output_file_path)

# Analyze the samples (e.g., by computing the empirical frequency of each keyword)
sample_freq = pd.value_counts(samples, normalize=True)
print(sample_freq)

def get_seed_documents():
    '''
        This method finds the labeled positives in the training set and collects them as the seed documents.
        These documents are later utilized for the purpose of retrieving the positive documents in the test set.

    :return: seed_documents, which is a list of LP documents in the training set (which are the seeds)
    '''
    # List to store documents where 'label_L40' is "positive/labeled"
    seed_documents = []
    file_path = '../data/pubmed-dse/L20/D000328.D008875.D015658/ordered_train.jsonl'
    # Open and read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Convert each line from JSON format to a Python dictionary
            document = json.loads(line)
            # Check if the 'label_L40' field equals "positive/labeled"
            if document.get('label_L40') == 'positive/labeled':
                # Add the document to the list if the condition is met
                seed_documents.append(document)
    return seed_documents

def reading_all_papers(ordered_input_file_path):
    all_papers = []
    with open(ordered_input_file_path, 'r') as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            all_papers.append({'Order': i + 1, 'Abstract': data['abstract']})
    return all_papers

test_path = r'../data/pubmed-dse/L20/D000328.D008875.D015658/ordered_test.jsonl'
all_papers = reading_all_papers(test_path)
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if
              token.isalpha() and token not in stopwords.words('english')]
    return ' '.join(tokens)

seed_corpus = get_seed_documents()
seed_texts = [doc['abstract'] for doc in seed_corpus]
processed_abstracts = [preprocess_text(abstract) for abstract in seed_texts]

# Step 3: Calculate TF-IDF scores
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_abstracts)


processed_abstracts = [preprocess_text(abstract) for abstract in seed_texts]

def get_orders_for_terms():
    df = pd.read_csv('../Monte_carlo_sampling_new/terms_with_probabilities.csv',encoding='utf-8')

    df_mcmc = pd.read_csv('MCMC_results/mcmc_iterations.csv',encoding='latin-1')
    merged_df = pd.merge(df_mcmc, df[['Term', 'order']], on='Term', how='left')
    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv('MCMC_results/ordered_mcmc_iterations.csv', index=False)

get_orders_for_terms()

def generating_distribution_plot():
    # Read the CSV file into a DataFrame
    df2 = pd.read_csv('MCMC_results/ordered_mcmc_iterations.csv')

    # Calculate the total count summed through all count values
    total_count = df2['Count'].sum()

    # Calculate the probability for each row by dividing the count by the total count
    df2['Probability'] = df2['Count'] / total_count

    # Save the modified DataFrame back to a new CSV file
    df2.to_csv('MCMC_results/ordered_mcmc_iterations.csv', index=False)

    # Call distribution_plot with both DataFrames
    distribution_plot(df, df2, 'skyblue', 'red', num_iterations)

    print(f'The converged samples are {converged_samples}')

generating_distribution_plot()

def get_test_set_positive_documents(test_path):
    """
        This method finds the truly positives in the testing set.
        These documents are later utilized for the calculation of precision and recall.

    :return: truly_positive_docs, which is a list of truly positives in the testing set.
    """
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
def term_probability_generation():
    distribution_path = 'MCMC_results/ordered_mcmc_iterations.csv'
    # Read the CSV file into a DataFrame
    data = pd.read_csv(distribution_path)

    # Extract 'Term' and 'Probability' columns
    terms = data['Term'].tolist()
    probabilities = data['Probability'].tolist()
    return terms, probabilities

terms, probabilities = term_probability_generation()
truly_positive_docs = get_test_set_positive_documents(test_path)

def calculate_performance_metrics(retrieved_documents, truly_positive_docs):
    '''
    In this method, it's implicitly assumed that the retrieved_documents and seed_documents are both list of strings.

    :param retrieved_documents: list of strings that represent the documents retrieved after each query.
    note that here we denote 'retrieved_documents' as the whole set of retrieved documents after n_mc iterations.
    :param seed_documents: list of strings that represent the set of seed documents defined.
    :return: seed recall and precision for each query
    '''

    retrieved_set = set(retrieved_documents)
    truly_positive_docs = set(truly_positive_docs)
    true_positive_set = retrieved_set & truly_positive_docs

    # Calculate seed recall
    true_positives = len(set(retrieved_documents) & set(truly_positive_docs))
    recall = true_positives / len(truly_positive_docs) if truly_positive_docs else 0

    # Calculate average seed precision
    precision = true_positives / len(retrieved_documents) if retrieved_documents else 0

    return recall, precision, true_positive_set

def reading_topn_percent(file_path,percentage):
    # Read the entire CSV file
    df_full = pd.read_csv(file_path)
    # Calculate the number of rows for each percentage
    num_rows = len(df_full)
    top_n_percent_rows = int(num_rows * percentage)
    # Extract the top 10%, 30%, and 50% of the content
    top_n_percent = df_full.head(top_n_percent_rows)
    # If needed, here's how to display the top 10% of the data, for example
    print(top_n_percent)
    output_path = f'top_percentages/top_{int(percentage*100)}_percent.csv'
    top_n_percent.to_csv(output_path,index=False)
    return output_path

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
    ax.set_xlabel('Parameter Combinations (n_mc, n_kw)')
    ax.set_ylabel('Scores')
    ax.set_title('MCMC: Precision and Recall by Parameter Combination')
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
    plt.savefig('MCMC_parameter_sweeping.png')  # Save the plot as a .png file
    plt.show()

def top_percentage_iteration_performance_visualization(precision_dict, recall_dict):
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
    ax.set_xlabel('Parameter Combinations (n_mc, n_kw, percentage)')
    ax.set_ylabel('Scores')
    ax.set_title('MCMC: Precision and Recall by Parameter Combination with thresholds')
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

    # Customize x-axis labels
    # plt.xticks(rotation=30, ha='right')  # Rotate the labels and align them to the right
    plt.tight_layout()  # Adjust the padding between and around subplots.
    # Other plotting code remains unchanged
    # Show the plot
    plt.savefig('MCMC_parameter_sweeping_with_thresholds.png')  # Save the plot as a .png file
    plt.show()

def printing_sorted_docs_with_frequencies(sorted_documents, n_mc, n_kw):
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
            csvwriter.writerow([doc, count, f'{normalized_freq:.4f}', abstract_text])

    # Function to calculate performance metrics
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
            f.write(
                f'\nIteration of N_MC = {n_mc}, N_kw = {n_kw} ==============================================================================================\n')

            # Perform the Monte Carlo sampling
            document_appearance_count = {}
            all_sampled_keywords = []  # Keep track of all sampled keywords for this parameter set

            iteration = 0
            for _ in tqdm(range(n_mc), desc='MCMC Sampling', unit='iteration'):
                iteration += 1
                # Here, we assume `monte_carlo_sampling` is a function that performs the sampling and returns the sampled keywords

                sampled_keywords = monte_carlo_sampling(terms, probabilities, n_kw)

                print(f'iteration {iteration}/{n_mc} : {sampled_keywords}')
                f.write(f'iteration {iteration}/{n_mc} : {sampled_keywords}\n')

                # Query the database with the sampled keywords, and calculate similarity for each iteration
                retrieved_documents = send_query_to_database(sampled_keywords, n_kw)
                document_sets.append(retrieved_documents)
                print(retrieved_documents)

                # Process the retrieved documents as required

                i = 0
                for doc in retrieved_documents:
                    i += 1
                    print(f'=> {i}th document retrieved:', doc['order'], doc['title'], doc['abstract'])
                    f.write(
                        f'=> {i}th document retrieved: Order: {doc["order"]}, Title: {doc["title"]}, Abstract: {doc["abstract"]}\n')

                # Depending on your implementation, you might want to keep track of document appearances
                for doc in retrieved_documents:
                    doc_id = doc['order']
                    document_appearance_count[doc_id] = document_appearance_count.get(doc_id, 0) + 1

                all_sampled_keywords.append(sampled_keywords)  # Store sampled keywords for later analysis

            # Register the documents based on sampled keywords
            for doc in processed_abstracts:
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
            printing_sorted_docs_with_frequencies(sorted_documents, n_mc, n_kw)

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

            # Extract the 'order' entries from the truly labeled positive documents
            truly_positive_text_orders = [doc['order'] for doc in truly_positive_docs]

            performance = calculate_performance_metrics(orders_of_retvd_docs, truly_positive_text_orders)
            # after calculating the performance metrics:
            recall, precision, true_positives_set = performance

            # write them to the output file to summarize its performance
            output_data.append(f"Parameter combination: N_MC={n_mc}, N_KW={n_kw}")
            output_data.append(f"Recall: {recall:.4f}")
            output_data.append(f"Precision: {precision:.4f}")
            output_data.append(f"True Positives is of size {len(true_positives_set)}(Order numbers): {sorted(true_positives_set)}")
            output_data.append(f"Truly Positive Documents is of size {len(truly_positive_docs)}(Order numbers): {sorted(truly_positive_text_orders)}")
            output_data.append(f"Retrieved Documents is of size {len(orders_of_retvd_docs)}(Order numbers): {sorted(orders_of_retvd_docs)}\n")
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

    # Convert the DataFrame to a string with a
    #
    # chart fo
    best_keywords_chart = df_best_keywords.to_string(index=False)

    # Define the file path where the output will be saved
    output_file_path = 'best_keywords_chart.csv'

    # Write the string to a text file
    with open(output_file_path, 'w',  encoding='utf-8') as file:
        file.write(best_keywords_chart)

    # After the loop, write all the output data to the file at once
    txt_output_file_path = 'iteration_metrics.txt'
    with open(txt_output_file_path, 'w', encoding='utf-8') as txtfile:  # 'w' for write mode, overwriting the file
        txtfile.write('\n'.join(output_data))

    print("All metrics and sets have been written to the file.")
    print(f'The size of best keywords is {len(best_keywords)}')

    return best_performance, best_params, best_keywords

def calculating_threshold_metrics_of_best_performance(best_params):
    # print(best_params)
    # Extract individual components
    n_mc = best_params['N_MC']
    n_kw = best_params['N_kw']

    # Extract the 'order' entries from the truly labeled positive documents
    truly_positive_text_orders = [doc['order'] for doc in truly_positive_docs]
    performance_path = f'parameter_sweeping/document_frequencies_nmc{n_mc}_nkw{n_kw}.csv'

    precision_dict = {}
    recall_dict = {}
    percentages = [0.1, 0.3, 0.5]
    for percentage in percentages:
        output_path = reading_topn_percent(performance_path, percentage)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(output_path)

        # Extract the 'Document Order' column and convert it into a list
        document_order_list = df['Document Order'].tolist()

        performance = calculate_performance_metrics(document_order_list, truly_positive_text_orders)
        recall, precision, true_positives_set = performance
        # write them to the output file to summarize its performance
        print(f"Parameter combination: N_MC={n_mc}, N_KW={n_kw} for top {percentage}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(
            f"True Positives is of size {len(true_positives_set)}(Order numbers): {sorted(true_positives_set)}")
        print(
            f"Truly Positive Documents is of size {len(truly_positive_docs)}(Order numbers): {sorted(truly_positive_text_orders)}")
        print(
            f"Retrieved Documents is of size {len(document_order_list)}(Order numbers): {sorted(document_order_list)}\n")
        print("")  # Add an empty string for a newline separator between entries

        precision_dict[(n_mc, n_kw, percentage)] = precision
        recall_dict[(n_mc, n_kw, percentage)] = recall
        top_percentage_iteration_performance_visualization(precision_dict, recall_dict)

_, best_params, best_keywords = monte_carlo_with_parameter_sweeping()
calculating_threshold_metrics_of_best_performance(best_params)
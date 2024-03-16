import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

import pandas as pd
import matplotlib.ticker as ticker


truly_pos_found_pos_path = 'candidate_list_evaluation/truly_pos_document_rankings_nkw5.csv'
lp_found_pos_path = 'candidate_list_evaluation/lp_document_rankings_nkw5.csv'

df_truly_pos = pd.read_csv(truly_pos_found_pos_path)
truly_pos_order = df_truly_pos['Document Order'].tolist()
truly_pos_order.sort()

df_lp = pd.read_csv(lp_found_pos_path)
lp_order = df_lp['Document Order'].tolist()
lp_order.sort()

print(truly_pos_order)
print(lp_order)

csv_file_path = 'BM25_scores_with_norm/lp_document_rankings_nkw5_normed.csv'
df_candidate = pd.read_csv(csv_file_path)
candidate_order = df_candidate['Document Order'].tolist()
candidate_order.sort()

def get_normalized_bm25_multi(csv_file, orders):
    normalized_scores = []
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row['Document Order']) in orders:
                normalized_scores.append(float(row['normalized_bm25_score']))

    # Return the list of normalized BM25 scores
    return normalized_scores

def get_normalized_df_multi(csv_file, orders):
    normalized_scores = []
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row['Document Order']) in orders:
                normalized_scores.append(float(row['Normalized Frequency']))

    # Return the list of normalized BM25 scores
    return normalized_scores

bm25_candidate = get_normalized_bm25_multi(csv_file_path, candidate_order)
df_candidate = get_normalized_df_multi(csv_file_path, candidate_order)
print(bm25_candidate)
print(df_candidate)

bm25_lp = get_normalized_bm25_multi(csv_file_path, lp_order)
df_lp = get_normalized_df_multi(csv_file_path, lp_order)
bm25_truly_pos = get_normalized_bm25_multi(csv_file_path, truly_pos_order)
df_truly_pos = get_normalized_df_multi(csv_file_path, truly_pos_order)

print(get_normalized_bm25_multi(csv_file_path, lp_order))
print(get_normalized_df_multi(csv_file_path, lp_order))
print(get_normalized_bm25_multi(csv_file_path, truly_pos_order))
print(get_normalized_df_multi(csv_file_path, truly_pos_order))

def calculate_average(data):
    if not data:
        return 0  # Handle case where the list is empty to avoid division by zero
    return sum(data) / len(data)

avg_df_truly_pos = calculate_average(df_truly_pos)
avg_bm25_truly_pos = calculate_average(bm25_truly_pos)

# Assuming you have x and y values for each category as numpy arrays or lists
candidates_x = df_candidate
candidates_y = bm25_candidate
relevant_found_x = df_lp
relevant_found_y = bm25_lp

print(relevant_found_x, relevant_found_y)

truly_pos_found_x = df_truly_pos
truly_pos_found_y = bm25_truly_pos

average_line_y = avg_bm25_truly_pos
average_line_x = avg_df_truly_pos
print(f'The average df of truly_pos documents is {average_line_y}')

plt.figure(figsize=(10, 5))

# Scatter plot for each category
plt.scatter(candidates_x, candidates_y, color='blue', label='Candidates')

plt.scatter(truly_pos_found_x, truly_pos_found_y, color='orange', marker='D', label='truly_pos FOUND')

plt.scatter(relevant_found_x, relevant_found_y, color='green', marker='s', label='Relevant FOUND')

plt.axhline(y=average_line_y, color='orange', linestyle='--', label='truly_pos FOUND Average BM25')

plt.axvline(x=average_line_x, color='orange', linestyle='--', label='truly_pos FOUND Average DF')


# Customize x-axis range
plt.xlim(0, 0.04)  # Replace x_min and x_max with the desired lower and upper bounds

# Labels and title
plt.xlabel('Normalized Document Frequency DF')
plt.ylabel('Normalized Document Relevance BM25')
plt.title('Document Relevance vs. Document Frequency')

# Legend
plt.legend()
plt.savefig('MC_BM25v.s.DF.png')  # Save the plot as a .png file
# Show the plot
plt.show()


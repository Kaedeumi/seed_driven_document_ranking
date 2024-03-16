import json

# Define the file paths
input_file_path = 'test.jsonl'
output_file_path = 'ordered_test.jsonl'

# Initialize the order counter
order_counter = 1

# Open the input file to read and output file to write
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    for line in input_file:
        # Parse the JSON object from each line in the input file
        data = json.loads(line.strip())

        # Add the 'order' field with the incremental order number
        data['order'] = order_counter
        order_counter += 1

        # Write the updated JSON object to the output file
        output_file.write(json.dumps(data) + '\n')

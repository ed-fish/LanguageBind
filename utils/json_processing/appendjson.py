import json

# File paths
file1_path = '/home/ef0036/Projects/LanguageBind/utils/all_plus_popsign.json'
file2_path = '/home/ef0036/Projects/LanguageBind/utils/bobs_manual_captions.json'
output_file_path = 'all_plus_popsign_bobsl_capt.json'

# Load the first JSON file (as a dictionary)
with open(file1_path, 'r') as file1:
    data1 = json.load(file1)

# Load the second JSON file (as a dictionary)
with open(file2_path, 'r') as file2:
    data2 = json.load(file2)

# Merge the dictionaries (data2 will overwrite data1 for any duplicate keys)
merged_data = {**data1, **data2}

# Count the total number of entries (keys) in the merged data
total_entries = len(merged_data)

# Write the merged data back to a new JSON file or overwrite one of the files
with open(output_file_path, 'w') as output_file:
    json.dump(merged_data, output_file, indent=4)

print(f"Successfully merged the files into {output_file_path}")
print(f"Total number of entries: {total_entries}")

import json
import sys

def count_entries(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            print(f"Number of entries: {len(data)}")
        elif isinstance(data, dict):
            print(f"Number of entries: {len(data)}")
        else:
            print("The JSON file does not contain a list or a dictionary.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_entries.py <json_file>")
    else:
        json_file = sys.argv[1]
        count_entries(json_file)

import pickle
import csv

def pkl_to_csv(input_pkl_file, output_csv_file):
    # Load data from pickle file
    with open(input_pkl_file, "rb") as pkl_file:
        data = pickle.load(pkl_file)

    # Check if data is a list of dictionaries
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("The data should be a list of dictionaries")

    # Get CSV column headers from the first dictionary
    headers = data[0].keys()

    # Write data to CSV file
    with open(output_csv_file, "w", newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

    print(f"Data has been successfully converted to {output_csv_file}")

# Example usage
pkl_to_csv("/data/christinefang/unified-model-editing/glue_eval/dataset/hellaswag.pkl", "hellaswag.csv")

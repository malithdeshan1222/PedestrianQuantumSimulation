import csv
import arff

def csv_to_arff(csv_file_path, arff_file_path, relation_name):
    """
    Converts a CSV file to ARFF format.

    :param csv_file_path: Path to the input CSV file
    :param arff_file_path: Path to save the output ARFF file
    :param relation_name: Name of the ARFF relation
    """
    # Read the CSV file
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = list(csv_reader)

    # Extract attributes and data from the CSV
    attributes = [(col, 'STRING') for col in data[0]]  # Defaulting to STRING type for simplicity
    data_rows = data[1:]  # Exclude headers

    # Create the ARFF structure
    arff_data = {
        'relation': relation_name,
        'attributes': attributes,
        'data': data_rows
    }

    # Write to ARFF file
    with open(arff_file_path, 'w') as arff_file:
        arff.dump(arff_data, arff_file)
    print(f"ARFF file saved to {arff_file_path}")

# Example usage
csv_file = "your_file.csv"  # Replace with your CSV file path
arff_file = "your_file.arff"  # Replace with your desired ARFF output file path
relation_name = "your_relation_name"  # Replace with the relation name for your ARFF file

csv_to_arff(csv_file, arff_file, relation_name)
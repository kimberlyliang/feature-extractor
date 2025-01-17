import csv
import glob
import os

def multiply_by_two(x):
    return int(x) * 2

def read_csv(input_file):
    rows = []
    with open(input_file, mode='r') as infile:
        reader = csv.reader(infile)

        for row in reader:
            multiplied_row = [multiply_by_two(value) for value in row]
            rows.append(multiplied_row)

    return rows

def write_csv(output_file, rows):
    with open(output_file, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)

def multiply_csvs(input_directory, output_directory):
    csv_files = glob.glob(os.path.join(input_directory, '*.csv'), recursive=False)
    for input_file in csv_files:
        filename = os.path.basename(input_file)
        output_filename = 'multiplied_' + filename
        output_file = os.path.join(output_directory, output_filename)

        rows = read_csv(input_file)
        write_csv(output_file, rows)

if __name__ == "__main__":
    base_directory = os.path.dirname(os.path.dirname(__file__))

    input_directory = os.getenv('INPUT_DIR', os.path.join(base_directory, 'data/input/'))
    output_directory = os.getenv('OUTPUT_DIR', os.path.join(base_directory, 'data/output/'))

    multiply_csvs(input_directory, output_directory)

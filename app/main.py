import boto3
import csv
import glob
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger()
s3 = boto3.client("s3")

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
        logger.info("Multiplying file: %s", input_file)
        filename = os.path.basename(input_file)
        output_filename = 'multiplied_' + filename
        output_file = os.path.join(output_directory, output_filename)

        rows = read_csv(input_file)
        write_csv(output_file, rows)

def download_files_from_s3(bucket, prefix, input_directory):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in response:
        return

    for obj in response["Contents"]:
        key = obj["Key"]
        local_filename = os.path.join(input_directory, os.path.basename(key))
        s3.download_file(bucket, key, local_filename)

def upload_files_to_s3(bucket, prefix, output_directory):
    for filename in os.listdir(output_directory):
        local_path = os.path.join(output_directory, filename)
        s3_key = f"{prefix}{filename}"
        s3.upload_file(local_path, bucket, s3_key)

if __name__ == "__main__":
    logger.info("Initiating Multiplication")

    base_directory = os.path.dirname(os.path.dirname(__file__))

    input_directory = os.getenv('INPUT_DIR', os.path.join(base_directory, 'data/input/'))
    output_directory = os.getenv('OUTPUT_DIR', os.path.join(base_directory, 'data/output/'))

    os.makedirs(input_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)

    bucket = os.getenv('S3_BUCKET', 'local-data-bucket')
    environment = os.getenv('ENVIRONMENT', 'LOCAL').upper()

    if environment != 'LOCAL':
        logger.info(f"Downloading files from s3://{bucket}/input/ to {input_directory}")
        download_files_from_s3(bucket, 'input/', input_directory)

    multiply_csvs(input_directory, output_directory)

    if environment != 'LOCAL':
        logger.info(f"Uploading files from {output_directory} to s3://{bucket}/output/")
        upload_files_to_s3(bucket, 'output/', output_directory)

    logger.info("Successfully Completed Multiplication")

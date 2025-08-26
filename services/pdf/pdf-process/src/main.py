import argparse
from utils import schema_generation, payment_schedule_extraction
import logging
import yaml
import argparse
import boto3
from botocore.exceptions import (
    BotoCoreError,
    NoCredentialsError,
    ClientError,
)
from botocore.client import BaseClient
logging.basicConfig(level=logging.INFO)
import json

# Boto 3 client Intialization
def s3_client() -> BaseClient:
    """
    Create and return a boto3 S3 client with error handling.

    This function attempts to initialize and return a boto3 S3 client. If AWS credentials are missing
    or a boto core error occurs during client creation, it logs the error and returns a descriptive string.

    Returns:
        BaseClient | str: A boto3 S3 client instance if successful, otherwise an error message string.

    Raises:
        None: All exceptions are caught and logged; function returns an error message string on failure.
    """
    try:
        return boto3.client("s3",verify=False)
    except NoCredentialsError:
        logging.error("AWS credentials not found.")
    except BotoCoreError as e:
        logging.error(f"An error occurred while creating the S3 client: {e}")
    return "An error occurred while creating the S3 client"


s3 = s3_client()
def get_secret():
 
    secret_name = "cpce-dev"
    region_name = "us-east-1"
 
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
 
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e
 
    secret = json.loads(get_secret_value_response['SecretString'])

    return secret
# Load config YAML
try:
    secret_cred = get_secret()
    config_bucket_name = secret_cred['CONFIG_BUCKET_NAME']
    config_file_key = secret_cred['CONFIG_KEY']

    response = s3.get_object(Bucket=config_bucket_name, Key=config_file_key)
    content = response["Body"].read().decode("utf-8")
    config = yaml.safe_load(content)

    logging.info("Successfully loaded config YAML.")
except (BotoCoreError, ClientError) as e:
    logging.error(f"Failed to fetch config file from S3: {e}")
    raise RuntimeError(
        f"Could not retrieve config YAML from S3 ({config_bucket_name}/{config_file_key}): {e}"
    ) from e
except yaml.YAMLError as e:
    logging.error(f"Failed to parse YAML content: {e}")
    raise RuntimeError(f"Could not parse YAML content from config file: {e}") from e
except Exception as e:
    logging.critical(f"Unexpected error loading config: {e}")
    raise RuntimeError(f"Unexpected error while loading config YAML: {e}") from e


def main(
    pdf_image_bucket_name: str,
    subfolder_prefix: str,
    pdf_table_extraction_output_bucket_name: str,
) -> str:
    """
    Executes the table extraction pipeline using LLM-based schema generation and extraction logic.

    This function initiates a table extraction process for PDF images stored in an S3 bucket.
    It generates a schema using an LLM, performs payment schedule extraction from the images,
    and stores the extracted table output in the specified S3 bucket. It also logs the process steps.

    Args:
        pdf_image_bucket_name (str): Name of the S3 bucket containing input PDF image files.
        subfolder_prefix (str): Subfolder path or prefix within the input bucket to process.
        pdf_table_extraction_output_bucket_name (str): Name of the S3 bucket where the extracted
            table output will be stored.

    Returns:
        str: The S3 key or filename of the extracted table output.

    Raises:
        None: Exceptions are either handled internally or allowed to propagate.
    """
    logging.info("Table extraction started...")
    schema_response = schema_generation(
        model_id=config["model_id"],
        data_model_bucket_name=config["data_model_bucket_name"],
        data_model_key=config["data_model_key"],
        budget_mapping_bucket_name=config["budget_mapping_bucket_name"],
        budget_mapping_key=config["budget_mapping_key"],
    )
    logging.info(schema_response)
    s3_output_filename = payment_schedule_extraction(
        model_id=config["model_id"],
        schema_response=schema_response,
        pdf_image_bucket_name=pdf_image_bucket_name,
        subfolder_prefix=subfolder_prefix,
        pdf_table_extraction_output_bucket_name=pdf_table_extraction_output_bucket_name,
    )
    logging.info("Table extraction process complete.")
    return s3_output_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="extract the payment schedule table from the images"
    )
    parser.add_argument(
        "--pdf-image-bucket-name",
        type=str,
        required=True,
        help="Bucket name for the input images of pdf that has payement schedule information.",
    )
    parser.add_argument(
        "--subfolder_prefix",
        type=str,
        required=True,
        help="Give path of specific file that need to be processed",
    )
    parser.add_argument(
        "--pdf-table-extraction-output-bucket-name",
        type=str,
        required=True,
        help="Bucket name where the output need to be saved.",
    )
    args = parser.parse_args()

    main(
        pdf_image_bucket_name=args.pdf_image_bucket_name,
        subfolder_prefix=args.subfolder_prefix,
        pdf_table_extraction_output_bucket_name=args.pdf_table_extraction_output_bucket_name,
    )
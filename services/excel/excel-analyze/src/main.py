"""It leverages utility functions for S3 interaction, 
dynamic module loading, sheet matching, and Excel structure analysis"""

import argparse
from utils import (
    get_s3_client,
    find_matching_sheet_from_s3,
    create_complex_excel_preview_from_s3,
    save_output_to_s3,
    metadata_extraction,
    get_dimension_headers,
    get_secret
)
import yaml

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(input_object_key: str, sheet_name: str) -> str:
    """
    Generate a structured preview of an Excel file and save it to an S3 bucket with an incrementing file name.

    Args:
        input_object_key (str): Key (path) of the Excel file in the input S3 bucket
        sheet_name (str): Type of sheet to process ('budget' or 'assumption')

    Returns:
        str: Success or error message
    """

    try:
        secret_cred = get_secret()
        config_bucket_name = secret_cred['CONFIG_BUCKET_NAME']
        config_file_key = secret_cred['CONFIG_KEY']
        s3 = get_s3_client()
        response = s3.get_object(Bucket=config_bucket_name, Key=config_file_key)
        content = response["Body"].read().decode("utf-8")
        config = yaml.safe_load(content)
        # Where the input file is present
        INPUT_BUCKET_NAME = config["input_bucket_name"]
        #header bucket
        header_bucket = config["data_model_bucket_name"]
        # header key
        header_key = config["data_model_key"]
        # Where the output bucket must be saved
        OUTPUT_BUCKET_NAME = config["output_bucket_name_excel_analyze"]
        # Bucket where the prompts are present
        llm_bucket_name_prompt = config["llm_bucket_name_prompt"]
        # Excel preview file
        file_key = config["file_key_excel_preview"]
        # .py file where all the sheet names are present
        sheet_list = config["sheet_name_list_file"]
    except Exception as e:
        logger.error(f"Error loading configuration from S3: {e}")
        raise RuntimeError(
            "Failed to load configuration from S3. Please check the config bucket, file key, and file contents."
        ) from e



    try:
        logger.info(f"Processing file: {input_object_key} for sheet type: {sheet_name}")

        try:
            SHEET_NAME = find_matching_sheet_from_s3(
                INPUT_BUCKET_NAME,
                input_object_key,
                sheet_name,
                llm_bucket_name_prompt,
                sheet_list,
            )
        except Exception as e:
            msg = f"Error finding matching sheet '{sheet_name}': {e}"
            logger.error(msg)
            return f"Error: Failed to find matching sheet - {e}"

        if not SHEET_NAME:
            return f"Error: No matching {sheet_name} sheet found in the Excel file."
        try:            
            # Call the function to extract metadata and update Excel
            meta_data_info = metadata_extraction(
                bucket_name=INPUT_BUCKET_NAME,
                key=input_object_key
                
            )

        except Exception as e:
            logger.warning(f"Failed to update metadata Excel: {e}")
        try:
            header = get_dimension_headers(sheet_name, header_bucket,header_key)
        except Exception as e:
            msg = f"Error getting the headers from JSON file for sheet '{sheet_name}': {e}"
            logger.error(msg)
            return f"Error getting the headers from JSON file for sheet: {e}"

        try:
            structure_message = create_complex_excel_preview_from_s3(
                INPUT_BUCKET_NAME,
                input_object_key,
                SHEET_NAME,
                sheet_name,
                header,
                meta_data_info,
                llm_bucket_name_prompt,
                file_key,
            )
        except Exception as e:
            msg = f"Error generating Excel preview for '{sheet_name}' sheet: {e}"
            logger.error(msg)
            return f"Error: Failed to generate Excel preview - {e}"

        try:
            result = save_output_to_s3(
                OUTPUT_BUCKET_NAME, input_object_key, structure_message, sheet_name
            )
        except Exception as e:
            msg = f"Error saving output to S3 for '{sheet_name}' sheet: {e}"
            logger.error(msg)
            return f"Error: Failed to save output to S3 - {e}"


        logger.info(result)
        return result

    except Exception as e:
        msg = f"Unexpected critical error in main function: {e}"
        logger.exception(msg)
        return f"Error: Unexpected critical error occurred - {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract structure from Excel in S3 and save to another S3 bucket."
    )
    parser.add_argument(
        "--input-key",
        type=str,
        required=True,
        help="Key (path) of the Excel file in the input S3 bucket",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        choices=["budget", "assumption", "timeline"],
        required=True,
        help="Type of sheet to process: 'budget' or 'assumption'",
    )
    args = parser.parse_args()

    result_message = main(args.input_key, args.sheet_name)
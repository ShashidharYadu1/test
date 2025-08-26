import argparse
import logging
from io import BytesIO
import yaml
from utils import (
    get_s3_client,
    fetch_object_from_s3,
    load_module_from_s3_in_memory,
    get_available_sheet_names,
    run_last_function,
    save_dataframe_to_s3,
    get_secret
)

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#secret manager
try:
    secret_cred = get_secret()
    config_bucket_name = secret_cred['CONFIG_BUCKET_NAME']
    config_file_key = secret_cred['CONFIG_KEY']
    s3 = get_s3_client()
    response = s3.get_object(Bucket=config_bucket_name, Key=config_file_key)
    content = response["Body"].read().decode("utf-8")
    config = yaml.safe_load(content)
    BUCKET_NAME = config["input_bucket_name"]
    SCRIPT_BUCKET_NAME = config["output_bucket_name_excel_analyze"]
    OUTPUT_BUCKET_NAME = config["excel_output_bucket"]
    OUTPUT_KEY_PREFIX = config["output_folder_name"]
    prompt_bucket_name = config["llm_bucket_name_prompt"]
    sheet_name_list = config["sheet_name_list_file"]

except Exception as e:
    logger.error(f"Error loading configuration from S3: {e}")
    raise RuntimeError("Failed to load configuration from S3. Please check the config bucket, file key, and file contents.") from e



def main(
    input_key: str,
    script_key: str,
    sheet_name: str,
):
    """
    Main function to process Excel files using dynamically loaded Python scripts.

    Args:
        input_key (str): Key of the Excel file in the input bucket.
        script_key (str): Key of the Python script in the script bucket.
        sheet_name (str): Type of sheet to process ('budget' or 'assumption')
    """
    try:
        logger.info(
            f"Fetching Python script from S3: {SCRIPT_BUCKET_NAME}/{script_key}"
        )
        script_content = fetch_object_from_s3(SCRIPT_BUCKET_NAME, script_key)
        if isinstance(script_content, str) and script_content.startswith("Error"):
            logger.error(script_content)
            return

        logger.info(f"Fetching Excel file from S3: {BUCKET_NAME}/{input_key}")
        # Input excel
        excel_bytes = fetch_object_from_s3(BUCKET_NAME, input_key)
        if isinstance(excel_bytes, str) and excel_bytes.startswith("Error"):
            logger.error(excel_bytes)
            return

        excel_io = BytesIO(excel_bytes)
        available_sheets = get_available_sheet_names(excel_io)


        module = load_module_from_s3_in_memory(prompt_bucket_name, sheet_name_list)

        # Choose appropriate sheet names based on sheet_name
        if sheet_name == "budget":
            target_sheets = module.EXPECTED_SHEET_NAMES
        elif sheet_name == "timeline":
            target_sheets = module.TIMELINE_SHEET_NAME
        elif sheet_name == "assumption":
            target_sheets = module.ASSUMPTIONS_SHEET_NAME
        else:
            logger.error(
                f"Invalid --sheet-type: {sheet_name}. Must be 'budget' or 'assumption' or 'timeline."
            )
            return

        matched_sheets = [sheet for sheet in target_sheets if sheet in available_sheets]

        if not matched_sheets:
            logger.warning(f"No matching sheets found for sheet type '{sheet_name}'.")
            return

                # Rename loop variable to avoid overwriting the original sheet_type
        if matched_sheets:
            sheet = matched_sheets[0]  # Take the first matched sheet only
            logger.info(f"Processing sheet: {sheet}")
            excel_io.seek(0)
            func_args = [excel_io, sheet, BUCKET_NAME, input_key]
            dfs = run_last_function(script_content, *func_args)

            for i, df in enumerate(dfs):
                base_name = input_key.split("/")[-1].replace(".xlsx", "").replace(" ", "_")
                suffix = f"{base_name}_{i + 1}" if len(dfs) > 1 else base_name

                # Use the original sheet_name from args for output path logic
                if sheet_name == "budget":
                    final_prefix = OUTPUT_KEY_PREFIX
                elif sheet_name == "assumption":
                    final_prefix = "assumption"
                elif sheet_name == "timeline":
                    final_prefix = "timeline"
                else:
                    final_prefix = "unknown"

                output_key = f"{final_prefix}/output_result_{suffix}.csv"

                save_result = save_dataframe_to_s3(df, OUTPUT_BUCKET_NAME, output_key)
                logger.info(save_result)
                
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Excel using generated script and store result in S3."
    )
    parser.add_argument(
        "--input_key",
        type=str,
        required=True,
        help="Key path of Excel file in input bucket",
    )
    parser.add_argument(
        "--script_key",
        type=str,
        required=True,
        help="Python script key in script bucket",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        required=True,
        choices=["budget", "assumption", "timeline"],
        help="Type of sheet to process: 'budget' or 'assumption'",
    )

    args = parser.parse_args()

    main(
        input_key=args.input_key,
        script_key=args.script_key,
        sheet_name=args.sheet_name,
    )
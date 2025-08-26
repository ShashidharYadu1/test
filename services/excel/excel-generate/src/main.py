
from typing import Dict, Any
import yaml
import argparse
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# Local imports
from utils import (
    load_module_from_s3_in_memory,
    invoke_headers_analysis,
    invoke_code_generation,
    extract_python_code,
    upload_python_code_to_s3,
    error_handler,
    get_s3_client,
    get_secret
)


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(
    preview_key: str, error_key: str, previous_code_key: str, sheet_name: str
) -> Dict[str, Any]:
    
    secret_cred = get_secret()
    config_bucket_name = secret_cred['CONFIG_BUCKET_NAME']
    config_file_key = secret_cred['CONFIG_KEY']

    # Load config
    try:
        s3 = get_s3_client()
        response = s3.get_object(Bucket=config_bucket_name, Key=config_file_key)
        content = response["Body"].read().decode("utf-8")
        config = yaml.safe_load(content)

        ######################### LLM CONFIGURATION #########################

        # Model-specific settings
        model_id = config["model_id"]
        anthropic_version = config["anthropic_version"]
        max_tokens = config["max_tokens"]
        temperature = config["temperature"]

        ####################### BUCKET CONFIGURATION #######################

        # Input/output bucket names
        metadata_bucket = config["input_bucket_name"]
        OUTPUT_BUCKET_NAME = config["output_bucket_name_excel_analyze"]
        DATA_MODEL_BUCKET_NAME = config["data_model_bucket_name"]
        DATA_MODEL_KEY = config["data_model_key"]
        PREVIEW_BUCKET_NAME = config[
            "output_bucket_name_excel_analyze"
        ]  # Same as output for preview

        ######################### PROMPT CONFIGURATION #######################

        # Prompt-related S3 paths
        prompt_bucket_name = config["llm_bucket_name_prompt"]
        prompt_key = config["prompts"]
        budget_prompt_path = config["budget_prompt"]
        assumption_prompt_path = config["assumption_prompt"]
        timeline_prompt_path = config["timeline_prompt"]


    except Exception as e:
        logger.error(f"Error loading configuration from S3: {e}")
        raise RuntimeError(
            "Failed to load configuration from S3. Please check the config bucket, file key, and file contents."
        ) from e






    # Determine prompt_type based on sheet name
    try:
        module = load_module_from_s3_in_memory(prompt_bucket_name, prompt_key)

        if sheet_name == "budget":
            prompt_type = "budget"
            code_gen_text = module.CODE_GEN
            header_info_text = module.HEADER_INFO
        elif sheet_name == "assumption":
            prompt_type = "assumption"
            code_gen_text = module.ASSUMPTION_CODE_GEN
            header_info_text = module.ASSUMPTION_IMP_INFO
        elif sheet_name == "timeline":
            prompt_type = "timeline"
            code_gen_text = module.TIMELINE_CODE_GEN
            header_info_text = module.TIMELINE_IMP_INFO
        else:
            raise ValueError(
                "Invalid --sheet-name. Must be 'budget' or 'assumption' or 'timeline'."
            )
    except Exception as e:
        logger.error(f"Error determining prompt type: {e}")
        return {"error": str(e)}

    # try:
    #     budget_png = encode_image_from_s3(DATA_MODEL_BUCKET_NAME, DATA_MODEL_KEY)
    # except Exception as e:
    #     logger.error(f"Error encoding data model image: {e}")
    #     return {"error": str(e)}

    try:
        s3_client = get_s3_client()
    except Exception as e:
        logger.error(f"Error initializing S3 client: {e}")
        return {"error": str(e)}

    try:
        response = s3_client.get_object(Bucket=PREVIEW_BUCKET_NAME, Key=preview_key)
        excel_preview = response["Body"].read().decode("utf-8")
    except Exception as e:
        logger.error(f"Error fetching Excel preview from S3: {e}")
        return {"error": str(e)}

    # try:
    #     metadata = metadata_extraction(metadata_bucket, preview_key)
    #     if metadata:
    #         logger.info("Metadata found and will be used for header analysis.")
    #     else:
    #         logger.warning("No metadata found. Proceeding without metadata.")
    # except Exception as e:
    #     logger.error(f"Error extracting metadata: {e}")
    #     return {"error": str(e)}

    try:
        headers_info = invoke_headers_analysis(
            model_id=model_id,
            excel_preview=excel_preview,
            header_message=header_info_text,
            anthropic_version=anthropic_version,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt_type=prompt_type,
        )
    except Exception as e:
        logger.error(f"Error during header analysis: {e}")
        return {"error": str(e)}

    try:
        if error_key and previous_code_key:
            error_response = s3_client.get_object(
                Bucket=PREVIEW_BUCKET_NAME, Key=error_key
            )
            error_message = error_response["Body"].read().decode("utf-8")
            previous_code_content = (
                s3_client.get_object(Bucket=PREVIEW_BUCKET_NAME, Key=previous_code_key)[
                    "Body"
                ]
                .read()
                .decode("utf-8")
            )

            python_code_output = error_handler(
                model_id=model_id,
                excel_preview=excel_preview,
                headers_info=headers_info,
                previous_code=previous_code_content,
                error_encountered=error_message,
                anthropic_version=anthropic_version,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            python_code_output = invoke_code_generation(
                model_id=model_id,
                excel_preview=excel_preview,
                headers_info=headers_info,
                code_gen=code_gen_text,
                anthropic_version=anthropic_version,
                max_tokens=max_tokens,
                temperature=temperature,
                prompt_type=prompt_type,
            )
    except Exception as e:
        logger.error(f"Error during code generation or error handling: {e}")
        return {"error": str(e)}

    try:
        python_code = extract_python_code(python_code_output)
    except Exception as e:
        logger.error(f"Error extracting Python code: {e}")
        return {"error": str(e)}

    try:
        output_key = upload_python_code_to_s3(
            s3_client=s3_client,
            bucket_name=OUTPUT_BUCKET_NAME,
            preview_key=preview_key,
            python_code=python_code,
            sheet_type=sheet_name,
        )
    except Exception as e:
        logger.error(f"Error uploading Python code to S3: {e}")
        return {"error": str(e)}

    try:
        logger.info(
            f"Excel data processing completed successfully."
        )
        return {"output_key": output_key}
    except Exception as e:
        logger.error(f"Final logging or return block failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Excel data and generate Python code."
    )
    parser.add_argument(
        "--preview-key",
        type=str,
        required=True,
        help="S3 key for the Excel preview text file",
    )
    parser.add_argument(
        "--error-key", type=str, help="Key for error message (optional)"
    )
    parser.add_argument(
        "--previous-code-key",
        type=str,
        help="Key for previous code containing error (optional)",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        required=True,
        choices=["budget", "assumption", "timeline"],
        help="Sheet name to determine prompt type (budget or assumption or timeline). Required.",
    )

    args = parser.parse_args()

    result = main(
        preview_key=args.preview_key,
        error_key=args.error_key,
        previous_code_key=args.previous_code_key,
        sheet_name=args.sheet_name,
    )
import boto3
import logging
import base64
import os
import re
import json
from types import ModuleType
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError
from botocore.client import BaseClient
from bedrock_client import invoke_bedrock_model
from typing import Tuple, Dict, Any
import yaml
import importlib

import warnings
from urllib3.exceptions import InsecureRequestWarning

# Suppress before anything else
warnings.simplefilter('ignore', InsecureRequestWarning)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

secret_cred = get_secret()
config_bucket_name = secret_cred['CONFIG_BUCKET_NAME']
config_file_key = secret_cred['CONFIG_KEY']

def get_s3_client() -> BaseClient:
    """
    Create and return an S3 client using boto3.

    Returns:
        BaseClient : A boto3 S3 client object if successful

    Logs:
        - An error if AWS credentials are not found.
        - An error if there is a BotoCore-related exception while creating the client.
    """
    try:
        return boto3.client("s3",verify = False)
    except NoCredentialsError:
        logging.error("AWS credentials not found.")
        raise RuntimeError("AWS credentials not found.")
    except BotoCoreError as e:
        logging.error(f"An error occurred while creating the S3 client: {e}")
        raise RuntimeError(f"Failed to intialize s3 client due to BotocCore error: {e}")


def load_module_from_s3_in_memory(bucket_name: str, file_key: str) -> ModuleType:
    """
    Load a Python module from an S3 object in memory. Module name is derived from the file key.

    Args:
        bucket_name (str): S3 bucket name
        file_key (str): S3 key for the file (e.g., 'path/to/module.py')

    Returns:
        module: The loaded Python module

    Raises:
        RuntimeError: If there are issues fetching, decoding, or executing the module
    """
    try:
        # Derive module name from file_key
        base_name = os.path.basename(file_key)  #  'excel_preview_message.py'
        module_name = os.path.splitext(base_name)[0]  # 'excel_preview_message'

        s3 = get_s3_client()

        try:
            logger.info(
                f"Fetching module code from S3: Bucket={bucket_name}, Key={file_key}"
            )
            # Get the .py file from the s3 bucket for excel-preview or sheet name.
            obj = s3.get_object(Bucket=bucket_name, Key=file_key)
            data = obj["Body"].read().decode("utf-8")
        except Exception as e:
            msg = f"Failed to read or decode file from S3 (Bucket={bucket_name}, Key={file_key}): {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e
        try:
            logger.info(f"Compiling and executing module '{module_name}' in memory.")
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            module = importlib.util.module_from_spec(spec)
            exec(data, module.__dict__)
            return module
        except Exception as e:
            msg = (
                f"Failed to compile or execute module '{module_name}' from S3 code: {e}"
            )
            logger.error(msg)
            raise RuntimeError(msg) from e

    except Exception as e:
        msg = f"[UnexpectedError] Could not load module '{module_name}' from S3: {e}"
        logger.exception(msg)
        raise RuntimeError(msg) from e


def load_json_from_s3(bucket_name: str, file_key: str) -> Dict[str, Any]:
    """
    Loads a JSON file from the specified S3 bucket and key.

    Args:
        bucket_name: Name of the S3 bucket.
        file_key: Path to the file inside the bucket.

    Returns:
        A dictionary representing the JSON content.
    """

    try:
        s3 = get_s3_client()
    except Exception as e:
        raise RuntimeError(f"Failed to get S3 client: {e}")

    try:
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    except ClientError as e:
        raise RuntimeError(
            f"Failed to retrieve object from s3://{bucket_name}/{file_key}: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error while retrieving object: {e}")

    try:
        data = obj["Body"].read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to read or decode S3 object body: {e}")

    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON from S3 object: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during JSON decoding: {e}")


def get_prompts_from_s3(
    config_bucket_name: str, config_file_key: str
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Loads the configuration YAML file and three prompt JSONs from S3.

    Args:
        config_bucket_name: S3 bucket containing the config file.
        config_file_key: Path to the config YAML file inside the bucket.

    Returns:
        A tuple containing the budget_prompt, assumption_prompt, and timeline_prompt.
    """
    try:
        try:
            s3 = get_s3_client()
        except Exception as e:
            logger.error(f"Failed to get S3 client: {e}")
            raise RuntimeError("Failed to initialize S3 client.") from e

        try:
            response = s3.get_object(Bucket=config_bucket_name, Key=config_file_key)
        except ClientError as e:
            logger.error(f"Failed to retrieve config file from S3: {e}")
            raise RuntimeError(
                f"Failed to get config file from s3://{config_bucket_name}/{config_file_key}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error retrieving config file: {e}")
            raise RuntimeError(
                f"Unexpected error while retrieving config from S3: {e}"
            ) from e

        try:
            content = response["Body"].read().decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to read or decode config file: {e}")
            raise RuntimeError("Failed to read or decode config file from S3.") from e

        try:
            config = yaml.safe_load(content)
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML format in config file: {e}")
            raise RuntimeError("Failed to parse config YAML file.") from e
        except Exception as e:
            logger.error(f"Unexpected error parsing config file: {e}")
            raise RuntimeError(f"Unexpected error during YAML parsing: {e}") from e

        try:
            prompt_bucket_name = config["llm_bucket_name_prompt"]
            budget_prompt_path = config["budget_prompt"]
            assumption_prompt_path = config["assumption_prompt"]
            timeline_prompt_path = config["timeline_prompt"]
        except KeyError as e:
            logger.error(f"Missing required key in config: {e}")
            raise RuntimeError(f"Missing required key in config: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error extracting prompt paths: {e}")
            raise RuntimeError(
                f"Unexpected error while extracting prompt paths: {e}"
            ) from e

        try:
            budget_prompt = load_json_from_s3(prompt_bucket_name, budget_prompt_path)
        except Exception as e:
            logger.error(f"Failed to load budget prompt: {e}")
            raise RuntimeError(f"Failed to load budget prompt from S3: {e}") from e

        try:
            assumption_prompt = load_json_from_s3(
                prompt_bucket_name, assumption_prompt_path
            )
        except Exception as e:
            logger.error(f"Failed to load assumption prompt: {e}")
            raise RuntimeError(f"Failed to load assumption prompt from S3: {e}") from e

        try:
            timeline_prompt = load_json_from_s3(
                prompt_bucket_name, timeline_prompt_path
            )
        except Exception as e:
            logger.error(f"Failed to load timeline prompt: {e}")
            raise RuntimeError(f"Failed to load timeline prompt from S3: {e}") from e

        return budget_prompt, assumption_prompt, timeline_prompt

    except Exception as outer_e:
        logger.error(f"Failed to load configuration or prompt files from S3: {outer_e}")
        raise outer_e


prompt, assumption_prompt, timeline_prompt = get_prompts_from_s3(
    config_bucket_name, config_file_key
)

def extract_python_code(generated_output: str) -> str:
    """
    Extract Python code from generated output using regex.

    Args:
        generated_output (str): Full generated text from LLM

    Returns:
        str: Extracted Python code between ```python ``` delimiters

    Raises:
        ValueError: If no Python code is found in the generated output
        TypeError: If input is not a string
        RuntimeError: For any unexpected errors during extraction
    """
    logger.info("Attempting to extract Python code from generated output...")

    # Step 1: Validate input type
    if not isinstance(generated_output, str):
        msg = "Input must be a string."
        logger.error(msg)
        raise TypeError(msg)

    # Step 2: Search for Python code block
    try:
        code_match = re.search(r"```python(.*?)```", generated_output, re.DOTALL)
    except re.error as regex_error:
        msg = f"Regex error while extracting Python code: {regex_error}"
        logger.error(msg)
        raise RuntimeError(
            "Failed to parse code due to invalid regex."
        ) from regex_error
    except Exception as e:
        msg = f"Unexpected error during regex search: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e

    # Step 3: Check if match was found
    if code_match:
        try:
            extracted_code = code_match.group(1).strip()
            logger.info("Python code successfully extracted.")
            return extracted_code
        except IndexError as index_error:
            msg = "Failed to extract code content from match group."
            logger.error(f"{msg} Error: {index_error}")
            raise RuntimeError(msg) from index_error
    else:
        msg = "No Python code found in the generated output."
        logger.error(msg)
        raise ValueError(msg)

def invoke_headers_analysis(
    model_id: str,
    excel_preview: str,
    header_message: str,
    anthropic_version: str,
    max_tokens: int,
    temperature: float,
    prompt_type: str,
):
    logger.info(
        f"Invoking Bedrock model for headers analysis ({prompt_type} prompt)..."
    )

    try:
        if prompt_type not in ["budget", "assumption", "timeline"]:
            raise ValueError(f"Unsupported prompt_type '{prompt_type}' provided.")
    except Exception as e:
        raise RuntimeError(f"Error validating prompt_type: {e}") from e

    try:
        if prompt_type == "assumption":
            prompt_input = assumption_prompt["headers_messages"]
        elif prompt_type == "timeline":
            prompt_input = timeline_prompt["headers_messages"]
        else:
            prompt_input = prompt["headers_messages"]
    except Exception as e:
        raise RuntimeError(f"Error accessing prompt messages: {e}") from e

    logger.info("Injecting dynamic values into prompt content.")

    try:
        updated_content = []

        for item in prompt_input["content"]:
            if item.get("type") == "text" and "text" in item:
                original_text = item["text"]
                replaced_text = original_text

                # Track replacements
                replaced_excel = False
                # replaced_metadata = False
                replaced_header = False

                if "{excel_preview}" in replaced_text:
                    if excel_preview:
                        replaced_text = replaced_text.replace(
                            "{excel_preview}", excel_preview
                        )
                        
                        replaced_excel = True
                    logger.info(
                        f"Excel preview replacement: {'Success' if replaced_excel else 'Failed (empty or not found)'}"
                    )

                if "{header_message}" in replaced_text:
                    if header_message:
                        replaced_text = replaced_text.replace(
                            "{header_message}", header_message
                        )
                        replaced_header = True
                    logger.info(
                        f"Header message replacement: {'Success' if replaced_header else 'Failed (empty or not found)'}"
                    )

                # Only keep the text block if it's not empty after replacement
                if replaced_text.strip():
                    item["text"] = replaced_text
                    updated_content.append(item)
                else:
                    logger.warning(
                        "Skipping text block: Resulting text is empty after replacements."
                    )
            else:
                # Pass through other types (e.g., unsupported blocks)
                updated_content.append(item)

        # Update prompt input with cleaned/filled content
        prompt_input["content"] = updated_content

    except Exception as e:
        raise RuntimeError(f"Error injecting dynamic text into prompt: {e}") from e

    logger.info("Calling Bedrock model...")

    try:
        headers_info = invoke_bedrock_model(
            model_id=model_id,
            prompt=prompt_input,
            anthropic_version=anthropic_version,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        raise RuntimeError(f"Error invoking Bedrock model: {e}") from e

    logger.info("Headers analysis completed successfully.")
    return headers_info


def invoke_code_generation(
    model_id: str,
    excel_preview: str,
    headers_info: str,
    code_gen: str,
    anthropic_version: str,
    max_tokens: int,
    temperature: float,
    prompt_type: str,
):
    logger.info(f"Invoking Bedrock model for code generation ({prompt_type} prompt)...")

    try:
        if prompt_type not in ["budget", "assumption", "timeline"]:
            raise ValueError(f"Unsupported prompt_type '{prompt_type}' provided.")
    except Exception as e:
        raise RuntimeError(f"Error validating prompt_type: {e}") from e

    try:
        if prompt_type == "assumption":
            prompt_input = assumption_prompt["code_messages"]
        elif prompt_type == "timeline":
            prompt_input = timeline_prompt["code_messages"]
        else:
            prompt_input = prompt["code_messages"]
    except Exception as e:
        raise RuntimeError(f"Error accessing prompt messages: {e}") from e

    logger.info("Injecting dynamic values into prompt content.")

    try:
        for item in prompt_input["content"]:
            # Check if it's a text content block
            if item.get("type") == "text" or "text" in item:
                original_text = item.get("text", "")
                updated_text = original_text

                # Track replacements
                replaced_excel = False
                replaced_headers = False
                replaced_code = False

                # Attempt to replace {excel_preview}
                if "{excel_preview}" in updated_text:
                    if excel_preview:  # Only replace if the value is non-empty
                        updated_text = updated_text.replace(
                            "{excel_preview}", excel_preview
                        )
                        replaced_excel = True
                    logger.info(
                        f"Excel preview replacement: {'Success' if replaced_excel else 'Failed (empty or not found)'}"
                    )

                # Attempt to replace {headers_info}
                if "{headers_info}" in updated_text:
                    if headers_info:
                        updated_text = updated_text.replace(
                            "{headers_info}", headers_info
                        )
                        replaced_headers = True
                    logger.info(
                        f"Headers info replacement: {'Success' if replaced_headers else 'Failed (empty or not found)'}"
                    )

                # Attempt to replace {code_gen}
                if "{code_gen}" in updated_text:
                    if code_gen:
                        updated_text = updated_text.replace("{code_gen}", code_gen)
                        replaced_code = True
                    logger.info(
                        f"Code gen replacement: {'Success' if replaced_code else 'Failed (empty or not found)'}"
                    )

                # Only update the item if the resulting text is not empty
                if updated_text.strip():
                    item["text"] = updated_text
                else:
                    logger.warning(
                        "Text block became empty after replacements and will be left unchanged."
                    )

    except Exception as e:
        raise RuntimeError(f"Error injecting dynamic text into prompt: {e}") from e

    logger.info("Calling Bedrock model...")

    try:
        python_code_output = invoke_bedrock_model(
            model_id=model_id,
            prompt=prompt_input,
            anthropic_version=anthropic_version,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        raise RuntimeError(f"Error invoking Bedrock model: {e}") from e

    logger.info("Code generation completed successfully.")
    return python_code_output


def error_handler(
    model_id: str,
    excel_preview: str,
    headers_info: str,
    previous_code: str,
    error_encountered: str,
    anthropic_version: str,
    max_tokens: int,
    temperature: float,
):
    logger.info("Invoking Bedrock model for code generation with error message...")

    try:
        prompt_input = prompt["error_handler"]
    except Exception as e:
        raise RuntimeError(f"Failed to fetch error handler prompt: {e}")

    try:
        for item in prompt_input["content"]:
            if "text" in item:
                original_text = item["text"]
                updated_text = original_text

                # Track replacements
                replaced_excel = False
                replaced_headers = False
                replaced_code = False
                replaced_error = False

                # Replace {excel_preview}
                if "{excel_preview}" in updated_text:
                    if excel_preview:
                        updated_text = updated_text.replace(
                            "{excel_preview}", excel_preview
                        )
                        replaced_excel = True
                    logger.info(
                        f"Excel preview replacement: {'Success' if replaced_excel else 'Failed (empty or not found)'}"
                    )

                # Replace {headers_info}
                if "{headers_info}" in updated_text:
                    if headers_info:
                        updated_text = updated_text.replace(
                            "{headers_info}", headers_info
                        )
                        replaced_headers = True
                    logger.info(
                        f"Headers info replacement: {'Success' if replaced_headers else 'Failed (empty or not found)'}"
                    )

                # Replace {python_code}
                if "{python_code}" in updated_text:
                    if previous_code:
                        updated_text = updated_text.replace(
                            "{python_code}", previous_code
                        )
                        replaced_code = True
                    logger.info(
                        f"Python code replacement: {'Success' if replaced_code else 'Failed (empty or not found)'}"
                    )

                # Replace {error_found}
                if "{error_found}" in updated_text:
                    if error_encountered:
                        updated_text = updated_text.replace(
                            "{error_found}", error_encountered
                        )
                        replaced_error = True
                    logger.info(
                        f"Error message replacement: {'Success' if replaced_error else 'Failed (empty or not found)'}"
                    )

                # Only update if result is non-empty
                if updated_text.strip():
                    item["text"] = updated_text
                else:
                    logger.warning(
                        "Text block became empty after replacements and was not updated."
                    )

    except Exception as e:
        raise RuntimeError(f"Failed during prompt text substitution: {e}") from e

    try:
        python_corrected_code_output = invoke_bedrock_model(
            model_id=model_id,
            prompt=prompt_input,
            anthropic_version=anthropic_version,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to invoke Bedrock model: {e}")

    return python_corrected_code_output


def upload_python_code_to_s3(
    s3_client: BaseClient,
    bucket_name: str,
    preview_key: str,
    python_code: str,
    sheet_type: str,
) -> str:
    """
    Upload Python code to an S3 bucket with a unique key.
    """
    try:
        logger.info("Generating unique output key for S3...")
        base_key, _ = os.path.splitext(preview_key)

        if sheet_type == "assumption":
            output_key = f"python_script/{base_key}_assumption_script.py"
        elif sheet_type == "timeline":
            output_key = f"python_script/{base_key}_timeline_script.py"
        elif sheet_type == "budget":
            output_key = f"python_script/{base_key}_script.py"
        else:
            raise ValueError("Invalid sheet type provided.")

        counter = 1
        while True:
            try:
                s3_client.head_object(Bucket=bucket_name, Key=output_key)
                if sheet_type == "assumption":
                    output_key = f"python_script/{base_key}_assumption_script_{counter}.py"
                elif sheet_type == "timeline":
                    output_key = f"python_script/{base_key}_timeline_script_{counter}.py"
                elif sheet_type == "budget":
                    output_key = f"python_script/{base_key}_script_{counter}.py"
                counter += 1
            except s3_client.exceptions.ClientError as e:
                if "404" in str(e):
                    break
                else:
                    raise RuntimeError("Failed to verify S3 object existence.")

        logger.info(
            f"Excel data processing completed successfully. Code uploaded to S3: {output_key}"
        )
        s3_client.put_object(Bucket=bucket_name, Key=output_key, Body=python_code)
        return output_key

    except Exception:
        logger.error("Unable to upload Python code to S3.")
        raise RuntimeError("An error occurred while uploading Python code to S3.")

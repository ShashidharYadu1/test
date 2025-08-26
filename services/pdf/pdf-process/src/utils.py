import base64
import json
import pandas as pd
import io
import os
from PIL import Image
import base64
from collections import defaultdict
import yaml
from bedrock_client import invoke_bedrock_model
import logging
import boto3
from botocore.exceptions import (
    BotoCoreError,
    NoCredentialsError,
    PartialCredentialsError,
    ClientError,
)
from botocore.client import BaseClient
from io import BytesIO
from typing import List

logging.basicConfig(level=logging.INFO)


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
        return boto3.client("s3", verify=False)
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

# Load prompt JSON
try:
    config_bucket_name = config["config_bucket_name"]
    prompt_file_key = config["prompt_file_key"]

    response = s3.get_object(Bucket=config_bucket_name, Key=prompt_file_key)
    content = response["Body"].read().decode("utf-8")
    prompt = json.loads(content)

    logging.info("Successfully loaded prompt JSON.")
except (BotoCoreError, ClientError) as e:
    logging.error(f"Failed to fetch prompt file from S3: {e}")
    raise RuntimeError(
        f"Could not retrieve prompt JSON from S3 ({config_bucket_name}/{prompt_file_key}): {e}"
    ) from e
except json.JSONDecodeError as e:
    logging.error(f"Failed to parse JSON content: {e}")
    raise RuntimeError(f"Could not parse JSON content from prompt file: {e}") from e
except Exception as e:
    logging.critical(f"Unexpected error loading prompt: {e}")
    raise RuntimeError(f"Unexpected error while loading prompt JSON: {e}") from e


### configured LLM Parameter
model_id = config["model_id"]
anthropic_version = config["anthropic_version"]
max_tokens = config["max_tokens"]
temperature = config["temperature"]

### Configured bucket credentials
data_model_bucket_name = config["data_model_bucket_name"]
data_model_key = config["data_model_key"]
budget_mapping_bucket_name = config["budget_mapping_bucket_name"]
budget_mapping_key = config["budget_mapping_key"]
meta_data_bucket_name = config["meta_data_bucket_name"]

### Metadata_suffix configration
metadata_suffix = config["meta_data_suffix"]


## Image to base64 conversion function
def encode_image(image_stream: BytesIO) -> str:
    """
    Encode an image stream to a base64-encoded UTF-8 string.

    This function reads binary data from a BytesIO stream and converts it into a base64-encoded
    string, suitable for embedding in prompts.

    Args:
        image_stream (BytesIO): A stream containing the binary image data.

    Returns:
        str: Base64-encoded string representation of the image content.

    Raises:
        ValueError: If the image stream cannot be read or encoding fails.
    """
    try:
        image_bytes = image_stream.read()
        if not image_bytes:
            raise ValueError("Empty image stream.")
        encoded_str = base64.b64encode(image_bytes).decode("utf-8")
        return encoded_str
    except Exception as e:
        raise ValueError(f"Failed to encode image: {e}")


# Extracted table postprocessing
def table_postprocessing(extraction_lst: list) -> pd.DataFrame:
    """
    Post-process extracted table data into a structured DataFrame.

    This function takes a list of JSON-formatted strings representing extracted table data,
    parses and merges them, and returns a clean pandas DataFrame with indexed payment IDs.

    Args:
        extraction_lst (list): List of JSON strings, each representing a partial table extraction result.

    Returns:
        pd.DataFrame: A DataFrame containing the combined table rows with an added 'payment_id' column.

    Raises:
        None: All parsing is assumed to succeed; errors will raise naturally if input is malformed.
    """
    all_data = []

    for idx, tuple_item in enumerate(extraction_lst):
        try:
            logging.info("The Json is getting processed...")

            # Find the start of the JSON string
            json_start = tuple_item.find("{")

            # Extract the JSON string
            json_str = tuple_item[json_start:]

            if json_start == -1:
                raise ValueError(f"No JSON object found in tuple at index {idx}.")

            # Parse the JSON string
            json_data = json.loads(json_str)
            data = json.dumps(json_data, indent=2)
            logging.info(f"Extracted JSON:{data}")

            # Add the parsed data to our list
            all_data.append(json_data)

        except json.JSONDecodeError as e:
            logging.error(f"Invalid json: {e}")
            raise RuntimeError(f"Incomplete Json object: {e}")

    try:
        # Merging all dictionaries
        combined_dict = defaultdict(list)
        for d in all_data:
            for key, value in d.items():
                combined_dict[key].extend(value)  # Append values

        # Convert defaultdict to normal dict
        combined_dict = dict(combined_dict)

        # Ensure "rows" key exists
        if "rows" not in combined_dict or not isinstance(combined_dict["rows"], list):
            raise KeyError(
                "'rows' key not found or is not a list in the combined dictionary."
            )

        # Create DataFrame
        df = pd.DataFrame(combined_dict["rows"])
        df["payment_id"] = df.index
        df = df.reset_index(drop=True)
        logging.info("DataFrame created successfully")
        return df

    except Exception as e:
        logging.error(f"Failed during DataFrame creation:{e}")
        raise RuntimeError(f"Failed creating DataFrame:{e}")


# Schema Generation
def schema_generation(
    model_id: str,
    data_model_bucket_name: str,
    data_model_key: str,
    budget_mapping_bucket_name: str,
    budget_mapping_key: str,
) -> dict:
    """
    Generate a schema for table extraction using a language model and budget mapping reference.

    This function performs the following:
    - Downloads a budget mapping Excel file and a data model image from specified S3 buckets.
    - Parses the Excel file to extract relevant tabular data.
    - Encodes the image for model input.
    - Prepares a prompt by embedding both the tabular and image data.
    - Invokes a language model (e.g., Anthropic Claude via Bedrock) to generate a structured schema.

    Args:
        model_id (str): Identifier of the language model to be used for schema generation.
        data_model_bucket_name (str): Name of the S3 bucket containing the data model image.
        data_model_key (str): Key (file path) of the data model image in the S3 bucket.
        budget_mapping_bucket_name (str): Name of the S3 bucket containing the budget mapping Excel file.
        budget_mapping_key (str): Key (file path) of the Excel file in the S3 bucket.

    Returns:
        dict: The JSON response from the language model containing the generated schema.

    Raises:
        RuntimeError: If any of the following operations fail:
            - Downloading files from S3 (e.g., missing credentials, access issues).
            - Reading or processing the Excel file.
            - Encoding the image for prompt inclusion.
            - Preparing the prompt structure.
            - Invoking the language model and receiving a response.
    """
    # Mapping excel file from the s3
    s3 = s3_client()
    try:
        budget_mapping_response = s3.get_object(
            Bucket=budget_mapping_bucket_name, Key=budget_mapping_key
        )
        budget_mapping_path = budget_mapping_response["Body"].read()
    except NoCredentialsError:
        logging.error("AWS credentials not found.")
        raise RuntimeError(
            "AWS credentials not found. Please configure them correctly."
        )
    except PartialCredentialsError:
        logging.error("Incomplete AWS credentials provided.")
        raise RuntimeError("Incomplete AWS credentials. Check AWS configuration.")
    except ClientError as e:
        logging.error(f"ClientError accessing S3: {e}")
        raise RuntimeError(
            f"Failed to retrieve object from S3: {e.response['Error']['Message']}"
        )
    except Exception as e:
        logging.error(f"Unexpected error accessing S3: {e}")
        raise RuntimeError(f"Unexpected error accessing S3 object: {e}")

    # Data model image from s3
    s3 = s3_client()
    try:
        budget_data_model = s3.get_object(
            Bucket=data_model_bucket_name, Key=data_model_key
        )
        data_model_json = json.load(budget_data_model["Body"])
    except NoCredentialsError:
        logging.error("AWS credentials not found.")
        raise RuntimeError(
            "AWS credentials not found. Please configure them correctly."
        )
    except PartialCredentialsError:
        logging.error("Incomplete AWS credentials provided.")
        raise RuntimeError("Incomplete AWS credentials. Check AWS configuration.")
    except ClientError as e:
        logging.error(f"ClientError accessing S3: {e}")
        raise RuntimeError(
            f"Failed to retrieve object from S3: {e.response['Error']['Message']}"
        )
    except Exception as e:
        logging.error(f"Unexpected error accessing S3: {e}")
        raise RuntimeError(f"Unexpected error accessing S3 object: {e}")

    try:
        # Reading the excel mapping data and converting to json
        schema_df = pd.read_excel(
            budget_mapping_path, sheet_name="Payment_Schedule_pdf"
        )
        budget_mapping_reference = schema_df[:12].drop(
            ["Mapped", "Mapping Logic", "Field Type"], axis=1
        )
        budget_mapping_reference_json = budget_mapping_reference.to_json(
            orient="records", indent=2
        )
    except Exception as e:
        logging.error(f"Error processing Excel to JSON: {e}")
        raise RuntimeError(f"Excel parsing or formatting failed: {e}")

    # Adding the Image and Excel mapping to the prompt
    try:
        prompt_input = prompt["schema_generation_prompt"]
        for item in prompt_input["content"]:
            if "text" in item and "{data_model_json}" in item["text"]:
                item["text"] = item["text"].replace(
                    "{data_model_json}", str(data_model_json)
                )
            if "text" in item and "{budget_mapping_reference_json}" in item["text"]:
                item["text"] = item["text"].replace(
                    "{budget_mapping_reference_json}", budget_mapping_reference_json
                )
    except Exception as e:
        logging.error(f"Prompt preparation failed: {e}")
        raise RuntimeError(f"Prompt formatting error: {e}")

    try:
        # Adding system prompt
        system_prompt = prompt["schema_system_prompt"]
        # Invoke Bedrock
        schema_response = invoke_bedrock_model(
            model_id=model_id,
            system_prompt=system_prompt,
            prompt=prompt_input,
            anthropic_version=anthropic_version,
            max_tokens=max_tokens,
            temperature=temperature,
            file_name="schema_generation",
        )
        return schema_response

    except Exception as e:
        logging.exception("An unexpected error occurred during schema generation.")
        raise RuntimeError(f"Schema generation failed: {e}")


def search_s3_metadata(searchkey: str, metadata_bucket_name: str) -> str:
    """
    Search for a specific metadata file in an S3 bucket by filename suffix.

    This function searches through all objects in the given S3 bucket and returns the key of the file
    whose name ends with the specified search key suffix. It uses S3 pagination to handle large buckets.

    Args:
        searchkey (str): The base name to search for (excluding the suffix).
        metadata_bucket_name (str): Name of the S3 bucket to search within.

    Returns:
        str: S3 object key of the matched file if found, otherwise "File not found".

    Raises:
        RuntimeError: If AWS credentials are missing or invalid, or if any S3 client error occurs.
    """
    try:
        s3 = s3_client()
        paginator = s3.get_paginator("list_objects_v2")
        search_key_suffix = searchkey + metadata_suffix

        for page in paginator.paginate(Bucket=metadata_bucket_name):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(search_key_suffix):
                    return key
        return "File not found"

    except NoCredentialsError:
        logging.error("AWS credentials not found.")
        raise RuntimeError(
            "AWS credentials not found. Please configure them correctly."
        )
    except PartialCredentialsError:
        logging.error("Incomplete AWS credentials provided.")
        raise RuntimeError("Incomplete AWS credentials. Check AWS configuration.")
    except ClientError as e:
        logging.error(f"ClientError accessing S3: {e}")
        raise RuntimeError(
            f"Failed to retrieve object from S3: {e.response['Error']['Message']}"
        )
    except Exception as e:
        logging.error(f"Unexpected error accessing S3: {e}")
        raise RuntimeError(f"Unexpected error accessing S3: {e}")


def payment_schedule_extraction(
    model_id: str,
    schema_response: dict,
    pdf_image_bucket_name: str,
    subfolder_prefix: str,
    pdf_table_extraction_output_bucket_name: str,
) -> str:
    """
    Extracts payment schedule tables from PDF page images stored in S3 using a multi-step LLM-based approach.

    This function performs the following steps:
    - Lists and loads image files from a specified subfolder in an S3 bucket.
    - Encodes images and prepares prompts using a predefined schema and optional metadata.
    - Invokes a language model to extract tabular data from the images.
    - Optionally validates the LLM response using a secondary prompt.
    - Aggregates all validated results, converts them into a DataFrame, and uploads the final CSV to an output S3 bucket.

    Args:
        model_id (str): Identifier of the LLM to be used for table extraction and validation.
        schema_response (dict): Dictionary representing the schema that guides the data extraction process.
        pdf_image_bucket_name (str): Name of the S3 bucket containing PDF page images (as files).
        subfolder_prefix (str): S3 key prefix indicating the subfolder containing the PDF images.
        pdf_table_extraction_output_bucket_name (str): S3 bucket where the resulting CSV will be stored.

    Returns:
        str: S3 key (path) of the uploaded CSV file containing the extracted table data.

    Raises:
        RuntimeError: Raised when there are issues accessing S3, preparing prompts, invoking the model,
                      or uploading the CSV to S3.
    """
    # List to store matched PDFs for processing
    extraction_lst = []

    s3 = s3_client()
    try:
        image_file = s3.list_objects_v2(
            Bucket=pdf_image_bucket_name, Prefix=subfolder_prefix
        )
    except NoCredentialsError:
        logging.error("AWS credentials not found.")
        raise RuntimeError(
            "AWS credentials not found. Please configure them correctly."
        )
    except PartialCredentialsError:
        logging.error("Incomplete AWS credentials provided.")
        raise RuntimeError("Incomplete AWS credentials. Check AWS configuration.")
    except ClientError as e:
        logging.error(f"ClientError accessing S3: {e}")
        raise RuntimeError(
            f"Failed to retrieve object from S3: {e.response['Error']['Message']}"
        )
    except Exception as e:
        logging.error(f"Unexpected error accessing S3: {e}")
        raise RuntimeError(f"Unexpected error accessing S3 object: {e}")

    # Extract file keys
    if "Contents" not in image_file:
        logging.error(f"No files found in the folder: {subfolder_prefix}")
        raise RuntimeError(f"No file found in the subfolder {subfolder_prefix}")

    try:
        if isinstance(schema_response, dict):
            schema_response = json.dumps(schema_response)
        elif not isinstance(schema_response, str):
            raise TypeError("schema_response must be a dict or a JSON string.")
    except Exception as e:
        logging.error(f"Failed to prepare schema_response: {e}")
        raise RuntimeError("Schema response preparation failed.") from e

    for obj in image_file["Contents"]:
        try:
            key = obj["Key"]
            image_obj = s3.get_object(Bucket=pdf_image_bucket_name, Key=key)
            table_image = image_obj["Body"].read()
            image_stream = BytesIO(table_image)
            input_image = encode_image(image_stream)  # input image of table
        except Exception as e:
            logging.error(f"Error processing image {key}:{e}")
            raise RuntimeError(f"Unexpected error while loading image:{e}")

        try:
            file_name = key.split("/")[1]  # Get the folder name before the slash
            search_key = file_name.split("_pdf")[0]

            meta_data_key = search_s3_metadata(
                searchkey=search_key, metadata_bucket_name=meta_data_bucket_name
            )

            if meta_data_key == "File not found":
                logging.warning(
                    "Metadata not found, processing the file without metadata information..."
                )
                try:
                    prompt_input = prompt["extraction_prompt"]
                    validation_prompt = prompt["validation_prompt"]
                    for item in prompt_input["content"]:
                        if item["type"] == "image":
                            item["source"][
                                "data"
                            ] = input_image  # Replace with encoded image data
                        if "text" in item and "{schema_response}" in item["text"]:
                            item["text"] = item["text"].replace(
                                "{schema_response}", schema_response
                            )
                except Exception as e:
                    logging.error(f"Prompt preparation failed: {e}")
                    raise RuntimeError(f"Prompt formatting error: {e}")

                try:
                    system_prompt = prompt["extraction_system_prompt"]
                    table_extraction_result = invoke_bedrock_model(
                        model_id=model_id,
                        system_prompt=system_prompt,
                        prompt=prompt_input,
                        anthropic_version=anthropic_version,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        file_name=file_name,
                    )
                except Exception as e:
                    logging.exception(
                        "An unexpected error occurred during Table extraction."
                    )
                    raise RuntimeError(f"Table extraction failed: {e}")

                # Validation step one more LLM pass, to validate the extracted response.
                logging.info(
                    "Extracted response without metadata is getting validated..."
                )
                # Extract the JSON string from the first element of each tuple
                json_str = table_extraction_result[0]

                # Find the start of the JSON string
                json_start = json_str.find("{")

                # Extract the JSON string
                json_str = json_str[json_start:]
                try:
                    validation_prompt = prompt["validation_prompt"]
                    system_prompt = prompt["validation_system_prompt"]
                    for item in validation_prompt["content"]:
                        if item["type"] == "image":
                            item["source"][
                                "data"
                            ] = input_image  # Replace with encoded image data
                        if "text" in item and "{extracted_json}" in item["text"]:
                            item["text"] = item["text"].replace(
                                "{extracted_json}", str(json_str)
                            )
                        if "text" in item and "{schema_response}" in item["text"]:
                            item["text"] = item["text"].replace(
                                "{schema_response}", schema_response
                            )
                except Exception as e:
                    logging.error(f"Prompt preparation failed: {e}")
                    raise RuntimeError(f"Prompt formatting error: {e}")

                try:
                    validated_output = invoke_bedrock_model(
                        model_id=model_id,
                        system_prompt=system_prompt,
                        prompt=validation_prompt,
                        anthropic_version=anthropic_version,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        file_name=file_name,
                    )
                except Exception as e:
                    logging.exception(
                        f"An unexpected error occurred during validation of response:{e}"
                    )
                    raise RuntimeError(f"validation failed:{e}")
                # Final validated response for each image are appended to the list
                extraction_lst.append(validated_output)

            else:
                logging.info(
                    "Adding the metadata information to the prompt and extraction starting now..."
                )
                meta_data_prompt = prompt["meta_data_info"]
                try:
                    metadata_s3 = s3.get_object(
                        Bucket=meta_data_bucket_name, Key=meta_data_key
                    )
                    metadata_content = json.load(metadata_s3["Body"])
                    # Step 2: Parse metadata fields
                    try:
                        file_name = os.path.basename(key)
                        filename = file_name.split("_pdf")[0] + ".pdf"
                        vendor_name = (
                            metadata_content["supplier_rollup_name"][0][0]["label"]
                            if metadata_content.get("supplier_rollup_name")
                            else None
                        )
                        protocol_number = (
                            metadata_content["protocol_study_number"][0]["title"]
                            if metadata_content.get("protocol_study_number")
                            else None
                        )
                        currency = (
                            metadata_content["Currency"]["label"]
                            if metadata_content.get("Currency")
                            else None
                        )
                        contract_type = (
                            metadata_content["contract_document"]["label"]
                            if metadata_content.get("contract_document")
                            else None
                        )
                    except Exception as e:
                        msg = f"Error parsing metadata fields: {e}"
                        logging.error(msg)
                        raise RuntimeError(msg) from e

                    new_row = {
                        "file_name": filename,
                        "vendor_name": vendor_name,
                        "protocol_number": protocol_number,
                        "currency": currency,
                        "metadata_contract_type": contract_type,
                    }
                    metadata_content = json.dumps(new_row, indent=2)
                    logging.info(new_row)
                except Exception as e:
                    logging.error(f"Failed to load the metadata:{e}")
                    raise RuntimeError(
                        f"Unexpected error happened while trying to load the metadata:{e}"
                    )

                try:
                    prompt_input = prompt["extraction_prompt"]
                    # Read JSON data from file
                    prompt_input["content"] += meta_data_prompt["content"]
                    for item in prompt_input["content"]:
                        if item["type"] == "image":
                            item["source"][
                                "data"
                            ] = input_image  # Replace with encoded image data
                        if "text" in item and "{schema_response}" in item["text"]:
                            item["text"] = item["text"].replace(
                                "{schema_response}", schema_response
                            )
                        if "text" in item and "{metadata_content}" in item["text"]:
                            item["text"] = item["text"].replace(
                                "{metadata_content}", metadata_content
                            )
                except Exception as e:
                    logging.error(f"Prompt preparation failed: {e}")
                    raise RuntimeError(f"Prompt formatting error: {e}")

                try:
                    system_prompt = prompt["extraction_system_prompt"]
                    table_extraction_result = invoke_bedrock_model(
                        model_id=model_id,
                        system_prompt=system_prompt,
                        prompt=prompt_input,
                        anthropic_version=anthropic_version,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        file_name=file_name,
                    )
                except Exception as e:
                    logging.exception(
                        "An unexpected error occurred during Table extraction."
                    )
                    raise RuntimeError(f"Table extraction failed: {e}")

                # Validation step one more LLM pass, to validate the extracted response.
                logging.info("Extracted response with metadata is getting validated...")
                # Extract the JSON string from the first element of each tuple
                json_str = table_extraction_result[0]

                # Find the start of the JSON string
                json_start = json_str.find("{")

                # Extract the JSON string
                json_str = json_str[json_start:]
                logging.info("Extracted response is getting validated...")
                try:
                    validation_prompt = prompt["validation_prompt"]
                    for item in validation_prompt["content"]:
                        if item["type"] == "image":
                            item["source"][
                                "data"
                            ] = input_image  # Replace with encoded image data
                        if "text" in item and "{extracted_json}" in item["text"]:
                            item["text"] = item["text"].replace(
                                "{extracted_json}", str(json_str)
                            )
                        if "text" in item and "{schema_response}" in item["text"]:
                            item["text"] = item["text"].replace(
                                "{schema_response}", schema_response
                            )
                        if "text" in item and "{metadata_content}" in item["text"]:
                            item["text"] = item["text"].replace(
                                "{metadata_content}", metadata_content
                            )
                except Exception as e:
                    logging.error(f"Prompt preparation failed: {e}")
                    raise RuntimeError(f"Prompt formatting error: {e}")
                try:
                    system_prompt = prompt["validation_system_prompt"]
                    validated_output = invoke_bedrock_model(
                        model_id=model_id,
                        system_prompt=system_prompt,
                        prompt=validation_prompt,
                        anthropic_version=anthropic_version,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        file_name=file_name,
                    )
                except Exception as e:
                    logging.exception(
                        f"An unexpected error occurred during validation of response:{e}"
                    )
                    raise RuntimeError(f"validation failed:{e}")

                # Final validated response for each image are appended to the list
                extraction_lst.append(validated_output)

        except Exception as e:
            logging.info(f"Unexpected Error while processing the image:{e}")
            raise RuntimeError(f"Unexpected Error while processing the image:{e}")

    try:
        # Dataframe creation with the LLM response
        df = table_postprocessing(extraction_lst=extraction_lst)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)

        output_s3_key = "pdf/" + os.path.splitext(os.path.basename(key))[0] + ".csv"

        s3.put_object(
            Bucket=pdf_table_extraction_output_bucket_name,
            Key=output_s3_key,
            Body=csv_buffer.getvalue(),
        )
        logging.info(f"Successfully uploaded CSV to {output_s3_key}")
        return output_s3_key
    except (ValueError, TypeError) as e:
        logging.error(f"DataFrame processing failed: {e}")
    except ClientError as e:
        logging.error(f"Failed to upload CSV to S3: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during CSV generation/upload: {e}")

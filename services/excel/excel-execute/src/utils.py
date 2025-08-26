import boto3
import pandas as pd
import logging
from io import BytesIO
import ast
from botocore.exceptions import NoCredentialsError, BotoCoreError, ClientError
from botocore.client import BaseClient
import os
import importlib.util
from types import ModuleType
import traceback
import importlib
import yaml
import logging

import warnings
from urllib3.exceptions import InsecureRequestWarning
import json
# Suppress before anything else
warnings.simplefilter('ignore', InsecureRequestWarning)

import os
os.environ['SSL_CERT_FILE'] = '/dev/null'
os.environ['REQUESTS_CA_BUNDLE'] = ''

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
        return boto3.client('s3',verify = False)
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

# To get the python Script
def fetch_object_from_s3(bucket: str, key: str) -> bytes | str:
    """
    Fetches an object from the specified S3 bucket.

    Args:
        bucket (str): Name of the S3 bucket.
        key (str): Key of the object to fetch.

    Returns:
        bytes: Object data in bytes if successful.
        str: Error message if the operation fails.
    """
    try:
        s3 = get_s3_client()
        if isinstance(s3, str):
            return s3
        logger.info(f"Fetching object from S3: s3://{bucket}/{key}")
        response = s3.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()
    except ClientError as e:
        logger.error(f"Client error while fetching object: {e}")
        return f"Client error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error while fetching object: {e}")
        return f"Unexpected error: {e}"


def get_last_function_name(script_content: str) -> str:
    """
    Extracts the name of the last defined function in the provided script.

    Args:
        script_content (str): Python script content as a string.

    Returns:
        str: Name of the last function, or an error message.
    """
    try:
        tree = ast.parse(script_content)
        function_names = [
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        ]
        return function_names[-1] if function_names else "No function found"
    except Exception as e:
        logger.error(f"Failed to parse script: {e}")
        return f"Error: {e}"


def get_available_sheet_names(excel_io: BytesIO) -> list:
    """
    Returns the list of sheet names in an Excel file.

    Args:
        excel_io (BytesIO): In-memory Excel file.

    Returns:
        list: Sheet names found in the Excel file.
    """
    try:
        excel_io.seek(0)
        xls = pd.ExcelFile(excel_io)
        return xls.sheet_names
    except Exception as e:
        logger.error(f"Error reading Excel sheets: {e}")
        return []


def extract_imports_from_script(script: str) -> list:
    """
    Extracts a list of module names imported in a Python script.

    Args:
        script (str): Python script content.

    Returns:
        list: List of top-level imported module names.
    """
    try:
        tree = ast.parse(script)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
        return list(imports)
    except Exception as e:
        logger.error(f"Failed to extract imports: {e}")
        return []


def detect_third_party_modules(imports: list) -> list:
    """
    Identifies which modules are third-party (site-packages).

    Args:
        imports (list): List of imported modules.

    Returns:
        list: Third-party module names.
    """
    third_party = []
    for module in imports:
        try:
            spec = importlib.util.find_spec(module)
            if spec and spec.origin and "site-packages" in spec.origin:
                third_party.append(module)
        except Exception:
            continue
    return third_party

# to save error
def save_text_to_s3(
    content: str, bucket: str, key_prefix: str, script_key: str = None
) -> str:
    """
    Saves a plain text string to S3 using a filename derived from the script name.

    Args:
        content (str): Text content to upload.
        bucket (str): S3 bucket name.
        key_prefix (str): Folder path for saving the file (e.g., "error_logs/error").
        script_key (str): Optional. If provided, error filename will be based on this script name.

    Returns:
        str: Status message.
    """
    try:
        s3 = get_s3_client()
        if isinstance(s3, str):
            return s3

        # Determine base filename from script_key or fallback to default
        if script_key:
            script_base_name = os.path.splitext(os.path.basename(script_key))[0]
            prefix_dir = "/".join(key_prefix.split("/")[:-1]) + "/"
            error_key = f"{prefix_dir}error_{script_base_name}.txt"
        else:
            # Fallback behavior for generic errors
            error_key = f"{key_prefix}.txt"

        # Upload the error log
        s3.put_object(Bucket=bucket, Key=error_key, Body=content.encode("utf-8"))
        logger.info(f"Error log uploaded to s3://{bucket}/{error_key}")
        return f"Error log uploaded to S3 as {error_key}."

    except Exception as e:
        logger.error(f"Failed to save error log to S3: {e}")
        return f"Failed to upload error log: {e}"
try:
    s3 = get_s3_client()
    response = s3.get_object(Bucket=config_bucket_name, Key=config_file_key)
    content = response["Body"].read().decode("utf-8")
    config = yaml.safe_load(content)
    OUTPUT_BUCKET_NAME = config["output_bucket_name_excel_analyze"]

except Exception as e:
    logger.error(f"Error loading configuration from S3: {e}")
    raise RuntimeError("Failed to load configuration from S3. Please check the config bucket, file key, and file contents.") from e


def run_last_function(
    script_code: str,
    *args,
    error_log_bucket: str = OUTPUT_BUCKET_NAME,
) -> list:
    """
    Executes the last function from a Python script using an S3 path.
    Assumes args contains:
        [sheet_name, bucket_name, file_key]
    """
    try:
        # Parse arguments
        excel_io = args[0]
        sheet_name = args[1]
        bucket_name = args[2]
        file_key = args[3]

        # --- Step 1: Identify and validate third-party modules ---
        imports = extract_imports_from_script(script_code)
        third_party_modules = detect_third_party_modules(imports)
        allowed_third_party = {"pandas", "numpy", "boto3"}
        not_allowed = [
            mod for mod in third_party_modules if mod not in allowed_third_party
        ]
        logger.info(f"Third-party libraries detected in script: {third_party_modules}")
        if not_allowed:
            logger.error(
                f"Disallowed third-party libraries found: {not_allowed}. Process will stop."
            )
            raise ImportError(f"Disallowed third-party libraries used: {not_allowed}")

        # --- Step 2: Proceed and run last function ---
        last_func_name = get_last_function_name(script_code)
        if last_func_name.startswith("Error") or last_func_name == "No function found":
            logger.warning(last_func_name)
            return []

        exec_namespace = {}
        exec(script_code, exec_namespace)
        func = exec_namespace.get(last_func_name)
        if not callable(func):
            logger.error(f"Function '{last_func_name}' is not callable.")
            return []

        s3_url = f"s3://{bucket_name}/{file_key}"
        logger.info(
            f"Running function: {last_func_name} with S3 path: {s3_url}, Sheet: {sheet_name}"
        )
        result = func(s3_url, sheet_name)

        dfs = []
        if isinstance(result, tuple):
            df, error = result
            if isinstance(df, pd.DataFrame):
                dfs.append(df)
            if error:
                logger.warning(f"Function returned error: {error}")
        elif isinstance(result, pd.DataFrame):
            dfs.append(result)
        else:
            logger.warning("Function did not return a DataFrame.")
        return dfs

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error executing function from script: {e}")
        logger.error(error_trace)
        if error_log_bucket:
            # Pass the script_key to generate a script-specific error log
            script_base_name = os.path.basename(
                file_key
            )  # Or extract from metadata if available
            save_text_to_s3(
                error_trace,
                error_log_bucket,
                "error_logs/error",
                script_key=script_base_name,
            )
        return []


def save_dataframe_to_s3(df: pd.DataFrame, bucket: str, key: str) -> str:
    """
    Saves a pandas DataFrame to a CSV file and uploads it to S3.

    Args:
        df (pd.DataFrame): DataFrame to upload.
        bucket (str): Destination S3 bucket name.
        key (str): S3 key where the file will be saved.

    Returns:
        str: Status message indicating success or failure.
    """
    try:
        s3 = get_s3_client()
        if isinstance(s3, str):
            return s3

        base_key, ext = os.path.splitext(key)
        counter = 0
        current_key = key

        while True:
            try:
                s3.head_object(Bucket=bucket, Key=current_key)
                counter += 1
                current_key = f"{base_key}_{counter}{ext}"
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    break
                else:
                    logger.error(f"Error checking object existence: {e}")
                    return f"Error checking object existence: {e}"

        output_buffer = BytesIO()
        df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        s3.put_object(Bucket=bucket, Key=current_key, Body=output_buffer.getvalue())
        logger.info(f"Uploaded CSV to s3://{bucket}/{current_key}")
        return f"Uploaded to s3"
    except Exception as e:
        logger.error(f"Failed to save DataFrame to S3: {e}")
        return f"Failed to save to S3: {e}"
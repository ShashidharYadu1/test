import subprocess
import argparse
import re
from typing import Optional, List
from pathlib import Path
import logging
import yaml
import boto3
from botocore.exceptions import (
    BotoCoreError,
    NoCredentialsError,
    PartialCredentialsError,
    ClientError,
    ProfileNotFound,
)
from botocore.client import BaseClient
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_aws_session():
    """
    Create AWS session with SSO profile support
    """
    try:
        # profile_name = os.getenv('AWS_PROFILE', 'default')
        profile_name = "Default"
        logger.info(f"Attempting to use AWS profile: {profile_name}")
        
        session = boto3.Session(profile_name=profile_name)
        
        # Test the session by getting caller identity
        sts_client = session.client('sts')
        identity = sts_client.get_caller_identity()
        logger.info(f"Successfully authenticated as: {identity.get('Arn', 'Unknown')}")
        
        return session
        
    except ProfileNotFound as e:
        logger.warning(f"Profile '{profile_name}' not found: {e}")
        logger.info("Trying to use default credentials...")
        
        # Fallback to default session
        try:
            session = boto3.Session()
            # Test the session
            sts_client = session.client('sts')
            identity = sts_client.get_caller_identity()
            logger.info(f"Using default credentials. Authenticated as: {identity.get('Arn', 'Unknown')}")
            return session
        except Exception as fallback_error:
            logger.error(f"Default credentials also failed: {fallback_error}")
            raise
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code in ['UnrecognizedClientException', 'InvalidUserID.NotFound']:
            raise Exception(
                "AWS credentials are invalid or expired. Please run 'aws sso login' to refresh your session."
            )
        elif error_code == 'AccessDenied':
            raise Exception("Access denied. Check your IAM permissions.")
        else:
            raise Exception(f"AWS authentication error: {e}")
            
    except NoCredentialsError:
        raise Exception(
            "No AWS credentials found. Please configure AWS SSO:\n"
            "1. Run: aws configure sso\n"
            "2. Run: aws sso login"
        )
    except Exception as e:
        logger.error(f"Unexpected error during AWS authentication: {e}")
        raise


def get_secret():
    """
    Get secret from AWS Secrets Manager with SSO support
    """
    secret_name = "cpce-dev"
    region_name = "us-east-1"

    try:
        # Create session with SSO support
        session = create_aws_session()
        client = session.client(service_name="secretsmanager", region_name=region_name)
        
        logger.info(f"Retrieving secret: {secret_name}")
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        
        secret = json.loads(get_secret_value_response["SecretString"])
        logger.info("Successfully retrieved secret from AWS Secrets Manager")
        
        return secret
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'UnrecognizedClientException':
            raise Exception(
                "Invalid AWS credentials or expired session. Please run 'aws sso login' to refresh your session."
            )
        elif error_code == 'ResourceNotFoundException':
            raise Exception(f"Secret '{secret_name}' not found in AWS Secrets Manager")
        elif error_code == 'AccessDeniedException':
            raise Exception(
                f"Access denied to secret '{secret_name}'. Check IAM permissions for secretsmanager:GetSecretValue"
            )
        else:
            raise Exception(f"AWS Secrets Manager error: {e}")
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse secret as JSON: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error retrieving secret: {e}")


def s3_client() -> BaseClient:
    """
    Create and return a boto3 S3 client with SSO support and error handling.
    """
    try:
        session = create_aws_session()
        client = session.client("s3", verify=False)
        logger.info("Successfully created S3 client")
        return client
        
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        raise Exception(f"Could not create S3 client: {e}")


# Initialize S3 client
try:
    s3 = s3_client()
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {e}")
    s3 = None

# Load config YAML
config = None
config_bucket_name = None
config_file_key = None

try:
    logger.info("Loading configuration from AWS Secrets Manager...")
    secret_data = get_secret()
    config_bucket_name = secret_data["CONFIG_BUCKET_NAME"]
    config_file_key = secret_data["CONFIG_KEY"]
    
    if not s3:
        raise Exception("S3 client is not available")
    
    logger.info(f"Fetching config from S3: s3://{config_bucket_name}/{config_file_key}")
    response = s3.get_object(Bucket=config_bucket_name, Key=config_file_key)
    content = response["Body"].read().decode("utf-8")
    config = yaml.safe_load(content)

    logger.info("Successfully loaded config YAML from S3")
    
except (BotoCoreError, ClientError) as e:
    error_msg = f"Failed to fetch config file from S3"
    if config_bucket_name and config_file_key:
        error_msg += f" (s3://{config_bucket_name}/{config_file_key})"
    error_msg += f": {e}"
    
    logger.error(error_msg)
    raise RuntimeError(error_msg) from e
    
except yaml.YAMLError as e:
    error_msg = f"Failed to parse YAML content from config file: {e}"
    logger.error(error_msg)
    raise RuntimeError(error_msg) from e
    
except Exception as e:
    error_msg = f"Unexpected error loading config YAML: {e}"
    logger.critical(error_msg)
    raise RuntimeError(error_msg) from e


def get_project_root():
    """
    Traverse upwards to find the root project directory (contains 'excel-analyze' etc.).
    """
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "pdf-extract").exists() or (parent / "pdf").exists():
            return parent
    raise RuntimeError("Project root not found.")


def pdf_extract(pdf_input_key: str, page_no: Optional[List[int]]):
    """
    Runs pdf-extract using the input key and optional page number
    to extract the page that contains payment schedule information.
    """
    if not config:
        raise RuntimeError("Configuration not loaded. Cannot proceed with pdf_extract.")
    
    input_bucket_name = config["input_bucket_name_pdf_pipeline_extract"]
    output_bucket_name = config["output_bucket_name_pdf_pipeline_extract"]
    project_root = get_project_root()
    pdf_extract_path = project_root / "pdf-extract" / "src" / "main.py"
    
    cmd = [
        "python",
        str(pdf_extract_path),
        "--input-bucket-name",
        input_bucket_name,
        "--input-object-key",
        pdf_input_key,
        "--output-bucket-name",
        output_bucket_name,
    ]
    
    if page_no:
        page_arg = ",".join(map(str, page_no))
        cmd.extend(["--page-no", page_arg])  # Fixed: added missing --
    
    if not pdf_extract_path.exists():
        raise FileNotFoundError(f"pdf-extract main.py not found at: {pdf_extract_path}")
    
    logger.info(f"Executing: {' '.join(cmd)}")
    output_lines = []
    object_key = None
    
    # Set environment variables for subprocess to inherit AWS credentials
    env = os.environ.copy()
    if 'AWS_PROFILE' in os.environ:
        env['AWS_PROFILE'] = os.environ['AWS_PROFILE']
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env  # Pass environment variables
    )
    
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if not line:
            continue
        line = line.rstrip("\n")
        output_lines.append(line)
        logger.info(line)
        
        # Look for S3 object key in logs
        if "Saved the extracted pages as pdf to following s3:" in line:
            match = re.search(
                r"Saved the extracted pages as pdf to following s3:\s*(.*)", line
            )
            if match:
                object_key = match.group(1).strip()
                logger.info(f"Found object key: {object_key}")
    
    process.wait()
    return_code = process.returncode
    
    if return_code != 0:
        raise RuntimeError(
            f"pdf-extract failed with return code {return_code}. Output:\n"
            + "\n".join(output_lines)
        )
    
    if not object_key:
        raise RuntimeError("Could not extract S3 object key from pdf-extract logs.")
    
    return object_key


def pdf_convert(input_object_key: str):
    """
    Runs pdf-convert using the input key.
    """
    if not config:
        raise RuntimeError("Configuration not loaded. Cannot proceed with pdf_convert.")
    
    project_root = get_project_root()
    pdf_convert_path = project_root / "pdf-convert" / "src" / "main.py"
    input_bucket_name = config["input_bucket_name_pdf_pipeline_convert"]
    output_bucket_name = config["output_bucket_name_pdf_pipeline_convert"]
    
    if not pdf_convert_path.exists():
        raise FileNotFoundError(f"pdf-convert main.py not found at: {pdf_convert_path}")
    
    cmd = [
        "python",
        str(pdf_convert_path),
        "--input-bucket-name",
        input_bucket_name,
        "--input-object-key",
        input_object_key,
        "--output-bucket-name",
        output_bucket_name,
    ]
    
    # Set environment variables for subprocess
    env = os.environ.copy()
    if 'AWS_PROFILE' in os.environ:
        env['AWS_PROFILE'] = os.environ['AWS_PROFILE']
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )
    
    output_lines = []
    subfolder_prefix = None
    
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if not line:
            continue
        line = line.rstrip("\n")
        output_lines.append(line)
        logger.info(line)
        
        # Look for script key in logs
        match = re.search(r"s3:\s*(.*)", line)
        if match:
            subfolder_prefix = match.group(1).strip()
            logger.info(f"Found object key: {subfolder_prefix}")
    
    process.wait()
    return_code = process.returncode
    
    if return_code != 0:
        raise RuntimeError(
            f"pdf-convert failed with return code {return_code}. Output:\n"
            + "\n".join(output_lines)
        )
    
    if not subfolder_prefix:
        raise RuntimeError("Could not extract script key from pdf-convert logs.")
    
    logger.info(f"Extracted script key: {subfolder_prefix}")
    return subfolder_prefix


def pdf_process(subfolder_prefix: str):
    """
    Runs pdf-process using the provided subfolder prefix.
    """
    if not config:
        raise RuntimeError("Configuration not loaded. Cannot proceed with pdf_process.")
    
    project_root = get_project_root()
    pdf_process_path = project_root / "pdf-process" / "src" / "main.py"
    input_bucket_name = config["input_bucket_name_pdf_pipeline_process"]
    output_bucket_name = config["output_bucket_name_pdf_pipeline_process"]
    
    if not pdf_process_path.exists():
        raise FileNotFoundError(f"pdf-process main.py not found at: {pdf_process_path}")
    
    cmd = [
        "python",
        str(pdf_process_path),
        "--pdf-image-bucket-name",
        input_bucket_name,
        "--subfolder_prefix",
        subfolder_prefix,
        "--pdf-table-extraction-output-bucket-name",
        output_bucket_name,
    ]
    
    # Set environment variables for subprocess
    env = os.environ.copy()
    if 'AWS_PROFILE' in os.environ:
        env['AWS_PROFILE'] = os.environ['AWS_PROFILE']
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )
    
    output_lines = []
    saved_path = None
    
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if not line:
            continue
        line = line.rstrip("\n")
        output_lines.append(line)
        logger.info(line)
        
        # Look for script key in logs
        match = re.search(r"INFO:root:Successfully uploaded CSV to (.+)", line)
        if match:
            saved_path = match.group(1).strip()
            logger.info(f"Found object key: {saved_path}")
    
    process.wait()
    return_code = process.returncode
    
    if return_code != 0:
        raise RuntimeError(
            f"pdf-process failed with return code {return_code}. Output:\n"
            + "\n".join(output_lines)
        )
    
    if not saved_path:
        raise RuntimeError("Could not extract script key from pdf-process logs.")
    
    logger.info(f"Extracted script key: {saved_path}")
    return saved_path


def run_pdf_pipeline_logic(pdf_input_key: str, page_no: Optional[List[int]] = None):
    """
    This function contains the complete orchestrated logic for the PDF pipeline.
    It's refactored from the original main() to be called as a background task.
    """
    try:
        logger.info(f"Starting PDF pipeline for input: {pdf_input_key}")
        
        # Verify AWS credentials at the start
        try:
            session = create_aws_session()
            logger.info("AWS credentials verified successfully")
        except Exception as e:
            logger.error(f"AWS credential verification failed: {e}")
            raise
        
        # Step 1: Run pdf extract
        logger.info(f"STEP 1: PDF EXTRACT for {pdf_input_key}")
        object_key = pdf_extract(pdf_input_key, page_no)
        logger.info(f"PDF extract completed. Output object key: {object_key}")

        # Step 2: Run pdf convert
        logger.info(f"STEP 2: PDF CONVERT for {object_key}")
        subfolder_prefix = pdf_convert(input_object_key=object_key)
        logger.info(f"PDF convert completed. Output subfolder prefix: {subfolder_prefix}")
        
        # Step 3: Run pdf process
        logger.info(f"STEP 3: PDF PROCESS for {subfolder_prefix}")
        output_file = pdf_process(subfolder_prefix=subfolder_prefix)
        logger.info(f"PDF process completed. Final output file: {output_file}")

        logger.info("\nSUCCESS! PDF Pipeline completed successfully!")
        return output_file

    except Exception as e:
        logger.error(
            f"[PIPELINE-ERROR] A critical error occurred in the PDF pipeline for {pdf_input_key}: {e}",
            exc_info=True,
        )
        raise


# Health check function for the service
def check_aws_connectivity():
    """
    Check if AWS credentials are working properly
    """
    try:
        session = create_aws_session()
        sts_client = session.client('sts')
        identity = sts_client.get_caller_identity()
        return {
            "status": "ok",
            "identity": identity.get('Arn', 'Unknown'),
            "account": identity.get('Account', 'Unknown')
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    # Test the AWS connectivity
    connectivity = check_aws_connectivity()
    if connectivity["status"] == "ok":
        logger.info(f"AWS connectivity test passed. Identity: {connectivity['identity']}")
    else:
        logger.error(f"AWS connectivity test failed: {connectivity['error']}")
import subprocess
import re
from pathlib import Path
import logging
import boto3
import os
from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
    ProfileNotFound
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_aws_session():
    """
    Create AWS session with SSO profile support
    """
    try:
        # Try to use the specified profile (from environment or default)
        profile_name = os.getenv('AWS_PROFILE', 'default')
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


def get_project_root():
    """
    Traverse upwards to find the root project directory (contains 'excel-analyze' etc.).
    """
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "excel-analyze").exists() or (parent / "excel").exists():
            return parent
    raise RuntimeError("Project root not found.")


def run_excel_analyze(input_key, sheet_name):
    """
    Runs excel-analyze and extracts the output key from its logs.
    Returns the dynamically generated preview key.
    """
    try:
        # Verify AWS credentials before starting subprocess
        try:
            session = create_aws_session()
            logger.info("AWS credentials verified for excel-analyze")
        except Exception as e:
            logger.error(f"AWS credential verification failed: {e}")
            raise
        
        project_root = get_project_root()
        excel_analyze_path = project_root / "excel-analyze" / "src" / "main.py"
        cmd = [
            "python",
            str(excel_analyze_path),
            "--input-key",
            input_key,
            "--sheet-name",
            sheet_name,
        ]
        logger.info(f"Running excel-analyze: {' '.join(cmd)}")

        if not excel_analyze_path.exists():
            error_msg = f"excel-analyze main.py not found at: {excel_analyze_path}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Found excel-analyze at: {excel_analyze_path}")

        # Set environment variables for subprocess to inherit AWS credentials
        env = os.environ.copy()
        if 'AWS_PROFILE' in os.environ:
            env['AWS_PROFILE'] = os.environ['AWS_PROFILE']
        if 'AWS_DEFAULT_REGION' in os.environ:
            env['AWS_DEFAULT_REGION'] = os.environ['AWS_DEFAULT_REGION']

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env  # Pass environment variables
        )
        output_key = None
        output_lines = []

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if not line:
                continue
            line = line.rstrip("\n")
            output_lines.append(line)
            logger.info(line)

            if "Output saved successfully to S3:" in line:
                match = re.search(r"Key=(.+?)\s*(?=,|$)", line)
                if match:
                    output_key = match.group(1).strip()
                    logger.info(f"Found full output key: {output_key}")

        process.wait()
        return_code = process.returncode

        if return_code != 0:
            error_msg = (
                f"excel-analyze failed with return code {return_code}. Output:\n"
                + "\n".join(output_lines)
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if not output_key:
            error_msg = "Could not extract output key from excel-analyze logs."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Extracted preview key: {output_key}")
        return output_key

    except Exception as e:
        logger.error(f"Exception in run_excel_analyze: {e}", exc_info=True)
        raise


def run_excel_generate(preview_key, sheet_name, error_key=None, previous_code_key=None):
    """
    Runs excel-generate using the preview key from excel-analyze.
    Returns the generated script key.
    """
    try:
        # Verify AWS credentials before starting subprocess
        try:
            session = create_aws_session()
            logger.info("AWS credentials verified for excel-generate")
        except Exception as e:
            logger.error(f"AWS credential verification failed: {e}")
            raise
        
        project_root = get_project_root()
        excel_generate_path = project_root / "excel-generate" / "src" / "main.py"
        cmd = [
            "python",
            str(excel_generate_path),
            "--preview-key",
            preview_key,
            "--sheet-name",
            sheet_name,
        ]

        if error_key:
            cmd += ["--error-key", error_key]
        if previous_code_key:
            cmd += ["--previous-code-key", previous_code_key]

        logger.info(f"Running excel-generate: {' '.join(cmd)}")

        if not excel_generate_path.exists():
            error_msg = f"excel-generate main.py not found at: {excel_generate_path}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Found excel-generate at: {excel_generate_path}")

        # Set environment variables for subprocess to inherit AWS credentials
        env = os.environ.copy()
        if 'AWS_PROFILE' in os.environ:
            env['AWS_PROFILE'] = os.environ['AWS_PROFILE']
        if 'AWS_DEFAULT_REGION' in os.environ:
            env['AWS_DEFAULT_REGION'] = os.environ['AWS_DEFAULT_REGION']

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env
        )

        script_key = None
        output_lines = []

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if not line:
                continue
            line = line.rstrip("\n")
            output_lines.append(line)
            logger.info(line)

            if "Code uploaded to S3:" in line:
                match = re.search(r"Code uploaded to S3:\s*(.+\.py)", line)
                if match:
                    script_key = match.group(1).strip()
                    logger.info(f"Found script key: {script_key}")

        process.wait()
        return_code = process.returncode

        if return_code != 0:
            error_msg = (
                f"excel-generate failed with return code {return_code}. Output:\n"
                + "\n".join(output_lines)
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if not script_key:
            error_msg = "Could not extract script key from excel-generate logs."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Extracted script key: {script_key}")
        return script_key

    except Exception as e:
        logger.error(f"Exception in run_excel_generate: {e}", exc_info=True)
        raise


def run_excel_run(input_key, script_key, sheet_name):
    """
    Runs excel-run using the input key and generated script.
    Returns tuple (success: bool, output_key: str, error_key: str or None)
    """
    try:
        # Verify AWS credentials before starting subprocess
        try:
            session = create_aws_session()
            logger.info("AWS credentials verified for excel-run")
        except Exception as e:
            logger.error(f"AWS credential verification failed: {e}")
            raise
        
        project_root = get_project_root()
        excel_execute_path = project_root / "excel-execute" / "src" / "main.py"
        cmd = [
            "python",
            str(excel_execute_path),
            "--input_key",
            input_key,
            "--script_key",
            script_key,
            "--sheet-name",
            sheet_name,
        ]
        logger.info(f"Running excel-run: {' '.join(cmd)}")

        if not excel_execute_path.exists():
            error_msg = f"excel-execute main.py not found at: {excel_execute_path}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Found excel-execute at: {excel_execute_path}")

        # Set environment variables for subprocess to inherit AWS credentials
        env = os.environ.copy()
        if 'AWS_PROFILE' in os.environ:
            env['AWS_PROFILE'] = os.environ['AWS_PROFILE']
        if 'AWS_DEFAULT_REGION' in os.environ:
            env['AWS_DEFAULT_REGION'] = os.environ['AWS_DEFAULT_REGION']

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env  # Pass environment variables
        )

        output_lines = []
        csv_output_key = None
        error_key = None

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if not line:
                continue
            line = line.rstrip("\n")
            output_lines.append(line)
            logger.info(line)

            if ".csv" in line and (
                "uploaded" in line.lower() or "saved" in line.lower()
            ):
                csv_match = re.search(r"([^/\s]+\.csv)", line)
                if csv_match:
                    csv_output_key = csv_match.group(1)
                    logger.info(f"Found CSV output: {csv_output_key}")

            if "Error log uploaded to" in line:
                error_match = re.search(r"error_logs/(.+\.txt)", line)
                if error_match:
                    error_key = f"error_logs/{error_match.group(1)}"
                    logger.info(f"Found error key: {error_key}")

        process.wait()
        return_code = process.returncode

        if csv_output_key:
            logger.info(f"Excel execution successful! CSV output: {csv_output_key}")
            return True, csv_output_key, None
        elif error_key:
            logger.warning(f"Excel execution failed. Error logged to: {error_key}")
            return False, None, error_key
        else:
            if return_code == 0:
                logger.info("Excel execution completed (no specific output detected)")
                return True, None, None
            else:
                logger.error(f"Excel execution failed with return code {return_code}")
                return False, None, None

    except Exception as e:
        logger.error(f"Exception in run_excel_run: {e}", exc_info=True)
        raise


def run_excel_pipeline_logic(
    input_key: str,
    sheet_name: str,
    error_key: str = None,
    previous_code_key: str = None,
):
    """
    This function contains the complete orchestrated logic for the Excel pipeline.
    It's refactored from the original main() to be called as a background task.
    """
    try:
        logger.info(f"Starting Excel pipeline for input: {input_key}")
        
        # Verify AWS credentials at the start of the pipeline
        try:
            session = create_aws_session()
            logger.info("AWS credentials verified successfully for Excel pipeline")
        except Exception as e:
            logger.error(f"AWS credential verification failed: {e}")
            raise

        logger.info(f"STEP 1: ANALYZING EXCEL FILE for {input_key}")
        preview_key = run_excel_analyze(input_key, sheet_name)

        current_error_key = error_key
        current_code_key = previous_code_key
        retry_count = 0
        max_retries = 2

        while retry_count <= max_retries:
            attempt_num = retry_count + 1
            logger.info(
                f"ATTEMPT {attempt_num}/{max_retries + 1}: GENERATING AND EXECUTING CODE"
            )

            logger.info(f"\nStep 2.{attempt_num}: Generating Python script...")
            if current_error_key:
                logger.info(f"Using error key: {current_error_key}")
            if current_code_key:
                logger.info(f"Using previous code key: {current_code_key}")

            script_key = run_excel_generate(
                preview_key=preview_key,
                sheet_name=sheet_name,
                error_key=current_error_key,
                previous_code_key=current_code_key,
            )

            logger.info(f"\nStep 3.{attempt_num}: Executing generated script...")
            success, output_key, new_error_key = run_excel_run(
                input_key=input_key, script_key=script_key, sheet_name=sheet_name
            )

            if success:
                logger.info("SUCCESS! Excel Pipeline completed successfully!")
                if output_key:
                    logger.info(f"Final output: {output_key}")
                return output_key
            else:
                if retry_count >= max_retries:
                    logger.error(
                        f"\nMaximum retries ({max_retries}) reached. Excel Pipeline failed."
                    )
                    logger.error(f"Last error key: {new_error_key}")
                    raise RuntimeError(f"Excel pipeline failed after {max_retries + 1} attempts")
                else:
                    logger.warning(
                        f"\nAttempt {attempt_num} failed. Preparing retry {attempt_num + 1}..."
                    )
                    current_error_key = new_error_key
                    current_code_key = script_key
                    retry_count += 1
                    logger.info(f"Will retry with error key: {current_error_key}")
                    logger.info(f"Will use previous code key: {current_code_key}")

    except Exception as e:
        logger.error(
            f"[PIPELINE-ERROR] A critical error occurred in the Excel pipeline for {input_key}: {e}",
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


def verify_project_structure():
    """
    Verify that all required project components exist
    """
    try:
        project_root = get_project_root()
        required_paths = [
            project_root / "excel-analyze" / "src" / "main.py",
            project_root / "excel-generate" / "src" / "main.py",
            project_root / "excel-execute" / "src" / "main.py"
        ]
        
        missing_paths = []
        for path in required_paths:
            if not path.exists():
                missing_paths.append(str(path))
        
        if missing_paths:
            return {
                "status": "error",
                "missing_paths": missing_paths
            }
        
        return {
            "status": "ok",
            "project_root": str(project_root)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    # Test AWS connectivity and project structure
    print("=== Excel Pipeline Health Check ===")
    
    # Check AWS connectivity
    connectivity = check_aws_connectivity()
    if connectivity["status"] == "ok":
        logger.info(f"AWS connectivity test passed. Identity: {connectivity['identity']}")
    else:
        logger.error(f"AWS connectivity test failed: {connectivity['error']}")
    
    # Check project structure
    structure = verify_project_structure()
    if structure["status"] == "ok":
        logger.info(f"Project structure verified. Root: {structure['project_root']}")
    else:
        logger.error(f"Project structure check failed: {structure.get('error', 'Missing paths: ' + ', '.join(structure.get('missing_paths', [])))}")
    
    print("=== Health Check Complete ===")
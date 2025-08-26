import logging
from typing import List
import fitz
import os
from PIL import Image
import boto3
from botocore.exceptions import (
    BotoCoreError,
    NoCredentialsError,
    PartialCredentialsError,
    ClientError,
    EndpointConnectionError,
)
from botocore.client import BaseClient
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)


# Boto 3 client Initialization
def s3_client() -> BaseClient:
    try:
        return boto3.client("s3",verify=False)
    except NoCredentialsError:
        logging.error("AWS credentials not found.")
        raise RuntimeError("AWS credentials not found.")
    except BotoCoreError as e:
        logging.error(f"An error occurred while creating the S3 client: {e}")
        raise RuntimeError(f"Failed to intialize s3 client due to BotocCore error: {e}")


def pdf_to_image(
    input_bucket_name: str, input_object_key: str, output_bucket_name: str
) -> List[str]:
    """
    Convert a PDF file from an S3 bucket into an image and upload it to another S3 bucket.

    This function reads a PDF file from the specified input S3 bucket, converts each page into
    an image using the PyMuPDF (fitz) library, and uploads the generated PNG image to the
    specified output S3 bucket. It handles AWS credential errors, client errors, and processing
    exceptions robustly, and avoids overwriting existing files by checking and adjusting output keys.

    Args:
        input_bucket_name (str): Name of the S3 bucket where the input PDF file is stored.
        input_object_key (str): S3 object key (path) to the input PDF file.
        output_bucket_name (str): Name of the S3 bucket to store the output image(s).

    Returns:
        List[str]: List of S3 URI successfully generated and uploaded image from the PDF.

    Raises:
        RuntimeError: If there are issues with AWS credentials, S3 access, or PDF/image processing.
    """
    s3 = s3_client()
    try:
        response = s3.get_object(Bucket=input_bucket_name, Key=input_object_key)
        pdf_bytes = response["Body"].read()
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

    # Checking the file that are already present in output bucket
    try:
        response = s3.list_objects_v2(Bucket=output_bucket_name)
    except NoCredentialsError:
        logging.error("AWS credentials not found.")
        raise RuntimeError("Missing AWS credentials.")
    except EndpointConnectionError as e:
        logging.error(f"Network issue while connecting to S3: {e}")
        raise RuntimeError(f"Connection error to S3: {e}")
    except ClientError as e:
        logging.error(f"AWS Client error: {e.response['Error']['Message']}")
        raise RuntimeError(f"S3 client error: {e.response['Error']['Message']}")
    except Exception as e:
        logging.error(
            f"Unexpected error when listing objects in bucket '{output_bucket_name}': {e}"
        )
        raise RuntimeError(f"Unexpected S3 error: {e}")

    try:
        pdf_stream = BytesIO(pdf_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")  # pdf read
        total_pages = doc.page_count
        s3_path = []
        for page_num in range(total_pages):
            try:
                page = doc.load_page(page_num)
                pix = page.get_pixmap(
                    dpi=300
                )  # Render the pdf page to image using fitz
                img_bytes = pix.tobytes(
                    "png"
                )  # Encode the image into png file and return raw bytes using fitz
                # Checking the file that are already present in output bucket
                existing_keys = [obj["Key"] for obj in response.get("Contents", [])]
                pdf_filename = input_object_key.split("/")[1]
                output_object_key = (
                    f"{pdf_filename}/{pdf_filename.split('.')[0]}_image_{page_num}"
                )
                output_number = 0
                while f"{output_object_key}_{output_number}.png" in existing_keys:
                    output_number += 1
                output_key = f"{output_object_key}_{output_number}.png"
                buffer = BytesIO(img_bytes)  # Temporary memory binary stream
                buffer.seek(0)  # Cursor to 0
                s3.upload_fileobj(buffer, Bucket=output_bucket_name, Key=output_key)
                logging.info(f"s3:{output_object_key}")
                s3_path.append(f"s3://{output_bucket_name}/{output_key}")
            except Exception as e:
                logging.error(
                    f"Error processing page {page_num} of '{input_object_key}': {e}"
                )
                raise RuntimeError(
                    f"Unexpected error while processing page {page_num} of {input_object_key}:{e}"
                )
        return s3_path
    except Exception as e:
        logging.error(f"Unexpected error while processing pdf:{e}")
        raise RuntimeError(f"Unexpected pdf processing error:{e}")
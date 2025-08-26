import argparse
from typing import Optional, List
from utils import extract_pages_as_single_pdf, process_and_extract_pdfs
import logging

# Create logger
logging.basicConfig(level=logging.INFO)

def main(
    input_bucket_name: str,
    input_object_key: str,
    output_bucket_name: str,
    page_no: Optional[List[int]],
) -> str:
    """
    Extract specific pages or the entire payment schedule from a PDF stored in an S3 bucket,
    then save the extracted pages as a new PDF in a specified output S3 bucket.

    Args:
        input_bucket_name (str): Name of the S3 bucket containing the source PDF.
        input_object_key (str): Key (path) of the source PDF file in the input S3 bucket.
        output_bucket_name (str): Name of the S3 bucket where the output PDF will be saved.
        page_no (Optional[List[int]]): List of page numbers to extract; if None, processes the pdf with logic extraction.

    Returns:
        str: The S3 path to the newly saved PDF if extraction is successful,
             or an error message if required arguments are missing.
    """

    if not all([input_bucket_name, input_object_key, output_bucket_name]):
        logging.error("Missing required arguments.")
        raise RuntimeError("Check the input arguments.")

    logging.info("PDF extraction process started...")

    if page_no:
        output_path = extract_pages_as_single_pdf(
            input_bucket_name=input_bucket_name,
            input_object_key=input_object_key,
            output_bucket_name=output_bucket_name,
            page_numbers=page_no,
        )
    else:
        output_path = process_and_extract_pdfs(
            input_bucket_name=input_bucket_name,
            input_object_key=input_object_key,
            output_bucket_name=output_bucket_name,
        )

    return output_path


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Extract specific pages from a PDF.")
        parser.add_argument(
            "--input-bucket-name",
            type=str,
            required=True,
            help="Bucket name for the input PDF from S3.",
        )
        parser.add_argument(
            "--input-object-key",
            type=str,
            required=True,
            help="Object key for the input PDF from S3.",
        )
        parser.add_argument(
            "--output-bucket-name",
            type=str,
            required=True,
            help="Bucket name for the output PDF to S3.",
        )
        parser.add_argument(
            "--page-no",
            type=int,
            nargs="*",
            required=False,
            help="List of page numbers to extract (optional).",
        )
        
        args = parser.parse_args()

        # Validate input/output filenames
        if not args.input_object_key.lower().endswith(".pdf"):
            logging.error("Input object key must point to a PDF file.")
            raise RuntimeError("Input object key must point to a PDF file.")

        result = main(
            input_bucket_name=args.input_bucket_name,
            input_object_key=args.input_object_key,
            output_bucket_name=args.output_bucket_name,
            page_no=args.page_no,
        )

        logging.info("PDF extract process completed successfully.")


    except Exception as e:
            logging.exception("Unhandled error during PDF processing.")
            raise
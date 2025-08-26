import logging
from utils import pdf_to_image
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)


def main(input_bucket_name: str, input_object_key: str, output_bucket_name: str) -> str:
    """
    Main function to convert a PDF from S3 into base64-encoded images and log the process.

    This function serves as the entry point for converting a PDF stored in an S3 bucket
    to a list of base64-encoded images, storing the images in another S3 bucket,
    and logging the outcome of the process.

    Args:
        input_bucket_name (str): Name of the S3 bucket containing the input PDF.
        input_object_key (str): Key (path) of the PDF file in the input S3 bucket.
        output_bucket_name (str): Name of the S3 bucket where the output images will be stored.

    Returns:
        Optional[str]: A message indicating the result of the process — either a success
        message with the number of pages converted, or a failure message.

    Raises:
        None: All exceptions are handled internally and logged.
    """

    logging.info(f"Processing started converting the PDF: {input_object_key} to image")

    base64_images = pdf_to_image(
        input_bucket_name, input_object_key, output_bucket_name
    )

    if base64_images:
        logging.info(f"Successfully converted {len(base64_images)} pages to Base64.")
        logging.info(f"Base64 data saved to {output_bucket_name}")
        return "Successfully converted {len(base64_images)} pages to Base64."
    else:
        logging.error("Failed to process the PDF.")
        raise RuntimeError("Converting PDF to Image failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the extracted pdf to images")
    parser.add_argument(
        "--input-bucket-name",
        type=str,
        required=True,
        help="Bucket name for the input PDF that has payement schedule information.",
    )
    parser.add_argument(
        "--input-object-key",
        type=str,
        required=True,
        help="Object key for the input PDF that has payement schedule information.",
    )
    parser.add_argument(
        "--output-bucket-name",
        type=str,
        required=True,
        help="Bucket name for the output image to S3.",
    )

    args = parser.parse_args()

    main(
        input_bucket_name=args.input_bucket_name,
        input_object_key=args.input_object_key,
        output_bucket_name=args.output_bucket_name,
    )
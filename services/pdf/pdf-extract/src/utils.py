import os
from typing import Optional, List
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextLine, LTChar
import pdfplumber
import fitz
import yaml
from typing import Optional, List, Union
import boto3
from io import BytesIO
from botocore.exceptions import (
    BotoCoreError,
    NoCredentialsError,
    PartialCredentialsError,
    ClientError,
)
from botocore.client import BaseClient
import logging
import json
# Logging configuration for monitoring
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
# Importing the cofig from s3
# Load config YAML
s3 = s3_client()
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


# Logic-1 to identify the top header of the page containing the text 'Payment Schedule'.
def extract_payment_schedule_pages(
    input_bucket_name: str,
    input_object_key: str,
    search_text: Union[str, List[str]],
    y_threshold: Optional[int] = None,
) -> List[int]:
    """
    Extract page numbers from a PDF stored in S3 that contain a top-of-page header with 'Payment Schedule'.

    This function identifies page in the PDF where the specified text appears near the top
    of the page (based on the y0 coordinate), and returns their page number.

    Args:
        input_bucket_name (str): Name of the S3 bucket where the PDF is stored.
        input_object_key (str): Key (path) of the PDF file in the S3 bucket.
        search_text (Union[str, List[str]]): Keyword(s) to search for.
        y_threshold (Optional[int]): Y-coordinate threshold help in locating element from the bottom of the page.

    Returns:
        List[int]: A list of page numbers containing the specified header at top of the page.

    Raises:
        Exception: If extraction fails.
    """
    try:
        # To check the given search text is str and postprocess to be lowercase
        if isinstance(search_text, str):
            search_text_lower = [search_text.lower()]
        else:
            search_text_lower = [text.lower() for text in search_text]
        # Pdf file input from s3 bucket
        # Access the PDF file from S3
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

        try:
            pdf_stream = BytesIO(pdf_bytes)
        except Exception as e:
            logging.error(f"Failed to create PDF stream from bytes: {e}")
            raise RuntimeError(
                "Unable to create PDF stream from the provided byte content."
            ) from e

        payment_schedule_pages = []

        for page_number, page_layout in enumerate(extract_pages(pdf_stream), start=1):
            for element in page_layout:
                if isinstance(
                    element, (LTTextBox, LTTextLine)
                ):  # Check for text elements
                    text = element.get_text().strip().lower()
                    for search_text in search_text_lower:
                        if search_text in text:
                            if y_threshold is None or element.y0 > y_threshold:
                                payment_schedule_pages.append(page_number)
                                break  # Move to next page after finding the header
        return payment_schedule_pages

    except Exception as e:
        logging.error(f"Error in extracting payment schedule pages: {e}")
        raise RuntimeError(f"Failed to extract the payment schedule pages due to: {e}")


# Logic 2 to check the pages contain table or not
def check_page_has_table(
    input_bucket_name: str, input_object_key: str, page_number: int
) -> bool:
    """
    Check if a specific page in a PDF, contains table or not.

    This function opens a PDF file from an S3 bucket and checks if the given page number
    contains any table or not using pdfplumber function 'extract tables'.

    Args:
        input_bucket_name (str): Name of the S3 bucket containing the PDF.
        input_object_key (str): Key (path) to the PDF file in the S3 bucket.
        page_number (int): Page number to check for presence of table.

    Returns:
        bool: True if the page contains at least one table, False otherwise.

    Raises:
        Exception: error and return False.
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

    pdf_stream = BytesIO(pdf_bytes)

    if not isinstance(page_number, int):
        logging.error(
            "Expected a single page number (int), Check the input argument page_number"
        )
        raise TypeError("Page number must be an integer")

    try:
        with pdfplumber.open(pdf_stream) as pdf:
            if 1 <= page_number <= len(pdf.pages):  # Ensure valid page number
                page = pdf.pages[page_number - 1]  # pdfplumber uses 0-based index
                tables = page.extract_tables()  # Extract tables
                return len(tables) > 0  # True if table exists, False otherwise
            else:
                logging.warning(f"This page don't have the table: {page_number}")
                return False
    except Exception as e:
        logging.error(
            f"Table identification process failed for this {page_number}: {e}"
        )
        return False


# Logic 3 to compare the previous table and new table in the next pages have same length
def extract_table_header_length(
    input_bucket_name: str, input_object_key: str, pages: list[int]
) -> dict:
    """
    Extract the number of non-empty columns in the second row of tables from specified PDF pages stored in S3.

    This function processes given page numbers from a PDF file in an S3 bucket and extracts the length
    of the second row (header) of the first table found on each page.

    Args:
        input_bucket_name (str): Name of the S3 bucket containing the PDF.
        input_object_key (str): Key (path) of the PDF file in the S3 bucket.
        pages (list[int]): List of page numbers to inspect for table headers.

    Returns:
        dict: A dictionary where keys are page numbers and values are the number of non-empty columns in the table header.

    Raises:
       Exception: error and return an empty dictionary if function fails.
    """
    if not (isinstance(pages, list) and all(isinstance(p, int) for p in pages)):
        logging.error("Check the input argument pages")
        raise TypeError("Page number must be an list of integer")

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

    try:
        header_lengths = {}
        pdf_stream = BytesIO(pdf_bytes)
        with pdfplumber.open(pdf_stream) as pdf:
            for page_number in pages:
                if 1 <= page_number <= len(pdf.pages):  # Check page number are valid
                    page = pdf.pages[
                        page_number - 1
                    ]  # pdfplumber uses 0-based indexing
                    tables = page.extract_tables()

                    if tables and tables[0] and len(tables[0]) >= 1:
                        header_row = tables[0][1]  # First row of the first table
                        header_lengths[page_number] = len(
                            [col for col in header_row if col is not None]
                        )
                    else:
                        logging.error(f"No tables found on page {page_number}")
                        raise RuntimeError(
                            f"Table not found on this page {page_number}, to check the row length"
                        )
        return header_lengths

    except Exception as e:
        logging.error(f"Error in extracting table header length: {e}")
        return {}


# Logic 4 Identification of keyword is present on the page
def search_keyword_in_pdf_page(
    input_bucket_name: str, input_object_key: str, page_number: int, search_key: str
) -> bool:
    """
    Check if a keyword exists on a specific page of a PDF stored in S3.

    Args:
        input_bucket_name (str): Name of the S3 bucket.
        input_object_key (str): Key (path) to the PDF file in the S3 bucket.
        page_number (int): Page number to search (1-based index).
        search_key (str): Keyword to search for.

    Returns:
        bool: True if keyword found on the page, False otherwise.
    """
    # Validate input arguments early
    if not isinstance(page_number, int) or page_number < 1:
        logging.error(f"Invalid page number provided: {page_number}")
        raise ValueError("Page number must be an integer >= 1")

    if not isinstance(search_key, str) or not search_key.strip():
        logging.error(f"Invalid search_key provided: {search_key}")
        # If keyword is empty or only whitespace, no need to search; return False immediately
        raise ValueError(f"Invalid search key: {search_key}")

    # Initialize S3 client
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

    # Get the PDF file from S3
    file_stream = BytesIO(pdf_bytes)

    try:
        # Read PDF using pdfplumber
        with pdfplumber.open(file_stream) as pdf:
            if page_number < 1 or page_number > len(pdf.pages):
                logging.error(f"Invalid page number: {e}")
                raise ValueError(
                    f"Invalid page number. PDF has {len(pdf.pages)} pages."
                )

            page = pdf.pages[page_number - 1]  # pdfplumber uses 0-based index
            text = page.extract_text()

            if text and search_key.lower() in text.lower():
                logging.info(f"Keyword '{search_key}' found on page {page_number}.")
                return True
            else:
                logging.info(f"Keyword '{search_key}' not found on page {page_number}.")
                return False

    except Exception as e:
        logging.error(f"Error while extracting text from page {page_number}: {e}")
        raise RuntimeError(
            f"Error while searching for the key word from page {page_number}: {e}"
        )


## Manual functional flow pass the page number manually
def extract_pages_as_single_pdf(
    input_bucket_name: str,
    input_object_key: str,
    output_bucket_name: str,
    page_numbers: List[int],
) -> str:
    """
     Extract specified pages from a PDF stored in S3 and save them as a single new PDF in another S3 bucket.

    Args:
        input_bucket_name (str): Name of the S3 bucket containing the source PDF.
        input_object_key (str): Key (path) to the source PDF file in the S3 bucket.
        output_bucket_name (str): Name of the S3 bucket to store the extracted PDF.
        page_numbers (List[int]): List of 1-based page numbers to extract from the PDF.

    Returns:
        str: S3 path to the saved PDF if successful, or a message indicating skip or error.
    """
    logging.info(f"Extracting the respective pages:{page_numbers}")
    if not (
        isinstance(page_numbers, list) and all(isinstance(p, int) for p in page_numbers)
    ):
        logging.error("Check the input argument 'page_numbers'")
        raise TypeError("Page number must be an list of integer")

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

    try:
        pdf_stream = BytesIO(pdf_bytes)

        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        new_doc = fitz.open()

        total_pages = doc.page_count  # Get total number of pages
        logging.info("Total number of pages present in the pdf are:", total_pages)
        for page_num in page_numbers:
            if page_num < 1 or page_num > total_pages:
                logging.error(
                    f"Page number {page_num} is out of range. Stopping execution."
                )
                raise ValueError(
                    f"Invalid page number: {page_num}. Must be between 1 and {total_pages}."
                )

            logging.info(f"Extracting page {page_num} ...")
            new_doc.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)
            logging.info(f"Added page {page_num} to the new document.")

        # Save the new document in memory buffer
        output_stream = BytesIO()
        new_doc.save(output_stream)
        output_stream.seek(0)

        response = s3.list_objects_v2(Bucket=output_bucket_name)
        existing_keys = [obj["Key"] for obj in response.get("Contents", [])]
        output_object_key = f"{input_object_key.split('.pdf')[0]}_pdf-extracted-pages"
        output_number = 0
        while f"{output_object_key}_{output_number}.pdf" in existing_keys:
            output_number += 1
        output_key = f"{output_object_key}_{output_number}.pdf"
        s3.upload_fileobj(output_stream, Bucket=output_bucket_name, Key=output_key)
        logging.info(
            f"Saved the extracted pages as pdf to following s3:{output_key}"
        )
        new_doc.close()
        doc.close()
        return f"s3://{output_bucket_name}/{output_object_key}"  # Return the full path of the saved file
    except Exception as e:
        logging.error(f"An error occurred while extracting pages: {str(e)}")
        raise RuntimeError(f"An error occurred while extracting pages: {str(e)}")


## Logic integration
def process_and_extract_pdfs(
    input_bucket_name: str, input_object_key: str, output_bucket_name: str
) -> str:
    """
    Process a PDF stored in S3 to extract pages containing payment schedule tables
    and save them as a new PDF in another S3 bucket.

    Args:
        input_bucket_name (str): Name of the S3 bucket containing the source PDF.
        input_object_key (str): Key (path) to the source PDF file in the S3 bucket.
        output_bucket_name (str): Name of the S3 bucket to store the extracted PDF.

    Returns:
        str: S3 path to the saved PDF if successful, or a message indicating skip or error.
    """
    try:
        page_lst = []
        # Search text for logic 4
        search_text = config["logic4_text"]
        # Y_threshold for logic 1
        y_threshold = config["y_threshold_logic4"]

        # Identifying the First page of payment schedule table with logic 1
        starting_pages = extract_payment_schedule_pages(
            input_bucket_name=input_bucket_name,
            input_object_key=input_object_key,
            search_text=search_text,
            y_threshold=y_threshold,
        )

        if not starting_pages:
            logging.info(
                f"No relevant pages found in {input_object_key}, skipping further processing."
            )
            return f"No relevant page found in this pdf: {input_object_key}, skipped..."

        first_page = starting_pages[-1]

        page_lst.append(first_page)

        # Identifying the First page has table or not with logic 2
        if not check_page_has_table(input_bucket_name, input_object_key, first_page):
            logging.info(
                f"No table found on page {first_page} in {input_object_key}, skipping."
            )
            return (
                f"No table found on page {first_page} in {input_object_key}, skipping."
            )

        # length of the second row (header) - First page table
        first_page_columns = extract_table_header_length(
            input_bucket_name, input_object_key, [first_page]
        )
        next_page = first_page + 1
        # Checking next page has table or not
        while check_page_has_table(input_bucket_name, input_object_key, next_page):
            next_page_columns = extract_table_header_length(
                input_bucket_name, input_object_key, [next_page]
            )  # length of the second row (header) - second page
            search_key = "Total"  # Search key for logic 4
            # Check that either logic 3 or logic 4 on second page
            if (
                first_page_columns
                and next_page_columns
                and set(next_page_columns.values()) == set(first_page_columns.values())
            ):
                page_lst.append(next_page)
                next_page += 1
            elif search_keyword_in_pdf_page(
                input_bucket_name=input_bucket_name,
                input_object_key=input_object_key,
                page_number=next_page,
                search_key=search_key,
            ):
                page_lst.append(next_page)
                next_page += 1
            else:
                break

        logging.info(
            f"These are the pages identified that have the payment scheudle information:{page_lst}"
        )
        if not page_lst or not all(
            isinstance(page, int) and page > 0 for page in page_lst
        ):
            logging.error(
                f"Extraction Failed for this pdf:{input_object_key}, The extracted page list are: {page_lst}"
            )
            return f"Failed pdf: {input_object_key}"

        # To save the new pdf with extracted page number that have payment scheduel information.
        extract_pages_as_single_pdf(
            input_bucket_name, input_object_key, output_bucket_name, page_lst
        )
        logging.info(f"PDF saved to : {output_bucket_name}")
        return f"The extraction of {output_bucket_name} saved to s3."

    except Exception as e:
        logging.error(
            f"Unexpected error occured while processing {input_object_key}: {str(e)}"
        )
        raise RuntimeError(
            f"An error occurred while processing {input_object_key}: {str(e)}"
        )
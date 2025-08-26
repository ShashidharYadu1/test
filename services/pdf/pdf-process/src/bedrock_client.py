import boto3
from typing import Tuple
import json
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
import usage
import time

usage_logger = usage.LLMUsageTracker()

boto3_config = Config(read_timeout=180)  # Read timeout configuration

def invoke_bedrock_model(
    model_id: str,
    system_prompt: str,
    prompt: dict,
    anthropic_version: str,
    max_tokens: int,
    temperature: int,
    file_name: str,
) -> Tuple[str, dict]:
    """
    Invoke an Amazon Bedrock-hosted LLM with the given prompt and parameters.

    This function sends a structured prompt to a specified Bedrock language model using Anthropic's API format,
    and returns both the text response and the full model output.

    Args:
        model_id (str): Identifier of the Bedrock-hosted language model to invoke.
        system_prompt (str): The system-level instruction to guide the model's behavior.
        prompt (dict): Prompt payload structured as per the model's expected message format.
        anthropic_version (str): Version of the Anthropic API to use.
        max_tokens (int): Maximum number of tokens the model is allowed to generate.
        temperature (float): Sampling temperature for randomness in generation (e.g., 0.7).

    Returns:
        Tuple[str, dict]: A tuple containing the extracted response text (str) and the full model output (dict).

    Raises:
        RuntimeError: If the Bedrock invocation fails or response cannot be parsed.
    """
    try:
        client = boto3.client(service_name="bedrock-runtime", config=boto3_config)
        body = {
            "anthropic_version": anthropic_version,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop_sequences": [],
            "system": system_prompt,
            "messages": [prompt],
        }
        start_time = time.time()
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        end_time = time.time()
        response_time = end_time - start_time 
        results = json.loads(response["body"].read())
        response = results["content"][0]["text"]

        ## Logging usage track of each llm call
        usage_logger.track_llm_call(
            results, model="claude-3-5-sonnet-20241022", file_name=file_name, response_time = response_time
        )

        return response

    except (BotoCoreError, NoCredentialsError, ClientError) as e:
        raise RuntimeError(f"Failed to invoke Bedrock model: {str(e)}")
    except (KeyError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to parse model response: {str(e)}")
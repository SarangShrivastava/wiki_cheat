import hashlib
import json
from math import ceil
import os
import pickle
import random
import time
import uuid
from collections import defaultdict
from pathlib import Path

import logging

import numpy as np
import pandas as pd
import requests
import tiktoken
import yaml
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from tenacity import (RetryCallState, retry,  # for exponential backoff
                      retry_if_exception_type, stop_after_attempt,
                      stop_after_delay, wait_random_exponential)

_ = load_dotenv(find_dotenv())
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


logger = logging.getLogger(__name__)

def num_tokens(
    prompt: str,
) -> int:
    """_summary_

    Args:
        prompt (str): _description_

    Returns:
        int: _description_
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(prompt))
    return num_tokens


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=(stop_after_delay(1200) | stop_after_attempt(5)),
    after=log_after_retry,
    reraise=True,
)
def completions_with_backoff(**kwargs)->ChatCompletion:
    return client.chat.completions.create(**kwargs)


class OpenAIClientException(Exception):
    """Some type of client error."""


@retry(
    retry=retry_if_exception_type(OpenAIClientException),
    wait=wait_random_exponential(min=1, max=60),
    stop=(stop_after_delay(600) | stop_after_attempt(3)),
    reraise=True,
)
def chat(message="", history=None, model="gpt-4", response_format=None,seed=None) -> ChatCompletion:
    """_summary_

    Args:
        message (str, optional): _description_. Defaults to "".
        history (_type_, optional): _description_. Defaults to None.
        model (str, optional): _description_. Defaults to "gpt-4".

    Returns:
        _type_: _description_
    """
    history = history or [{"role": "system", "content": "You are a helpful assistant."}]
    logger.info(f"Chatting with {model} (content {num_tokens(message)} tokens).")
    t0 = time.time()
    chat_completion_kwargs = {
        'model':model,
        'messages':history + [{"role": "user", "content": message}],
        'temperature':0,
    }
    if response_format:
        chat_completion_kwargs['response_format'] = {'type':response_format}
    if seed is not None:
        chat_completion_kwargs['seed'] = seed
    response = completions_with_backoff(
        **chat_completion_kwargs
    )
    t1 = time.time()

    logger.info(f"Response {response.usage.json()} ({round(t1-t0,3)} sec) Finish Reason: {response.choices[0].finish_reason}.")

    return response

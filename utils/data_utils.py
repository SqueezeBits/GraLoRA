import re
import os
import json
import logging
from packaging import version
from typing import List, Union

def extract_response(output: str, assistant_start: str) -> str:
    """
    Extracts the response from the output string starting from the assistant_start string.
    Args:
        output (str): The output string containing the response.
        assistant_start (str): The starting string of the assistant's response.
    Returns:
        str: The extracted response from the output string.
    """
    assert assistant_start in output
    return output[output.rindex(assistant_start) + len(assistant_start):]

def read_jsonl(file):
    """
    Read a JSONL file.

    Args:
        file (str): The path to the JSONL file.

    Returns:
        List[dict]: A list of dictionaries, each representing a sample.
    """
    if not os.path.exists(file):
        return []

    with open(file, "r") as f:
        return [json.loads(line) for line in f]

def append_jsonl(file: str, data: Union[dict, List[dict]]):
    """
    Append data to a JSONL file.

    Args:
        file (str): The path to the JSONL file.
        data (Union[dict, List[dict]]): The data to append.
    """
    if not os.path.exists(file):
        with open(file, "w") as f:
            pass

    if isinstance(data, dict):
        data = [data]

    with open(file, "a") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

def write_jsonl(file: str, data: Union[dict, List[dict]]):
    """
    Write data to a JSONL file.

    Args:
        file (str): The path to the JSONL file.
        data (Union[dict, List[dict]]): The data to write.
    """
    if isinstance(data, dict):
        data = [data]

    with open(file, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

def check_conversation_format(sample, tokenizer):
    """
    Check if the conversation format is valid.

    Args:
        sample (dict): The conversation sample.
        tokenizer: The tokenizer object.

    Returns:
        bool: True if the conversation format is valid, False otherwise.
    """
    try:
        tokenizer.apply_chat_template(
            conversation=sample["openai_format"],
            tokenize=False
        )
    except Exception:
        return False

    if len([turn for turn in sample["openai_format"] if turn["role"] == "assistant" ]) == 0:
        # No assistant turns
        logging.warning("No assistant turns in the conversation.")
        return False

    return True

def find_substrings_between_include(s, substr1, substr2, greedy=False):
    """
    Find the positions of substrings between two given substrings in a string.

    Args:
        s (str): The input string.
        substr1 (str): The first substring.
        substr2 (str): The second substring.
        greedy (bool, optional): Whether to use greedy matching. Defaults to False.

    Returns:
        list: A list of tuples representing the start and end positions of the substrings.

    Note: 
        re.DOTALL matches all characters, including newlines.
    """
    if greedy:
        pattern = re.escape(substr1) + r'(.*)' + re.escape(substr2)
    else:
        pattern = re.escape(substr1) + r'(.*?)' + re.escape(substr2)
    positions = []

    for match in re.finditer(pattern, s, re.DOTALL):
        start = match.start()
        end = match.end()
        content = match.group(0)
        positions.append((start, end))

    return positions

def find_substrings_between_exclude(s, substr1, substr2, greedy=False):
    """
    Find the positions of substrings between two given substrings in a string.

    Args:
        s (str): The input string.
        substr1 (str): The first substring.
        substr2 (str): The second substring.
        greedy (bool, optional): Whether to use greedy matching. Defaults to False.

    Returns:
        list: A list of tuples representing the start and end positions of the substrings.

    Note: 
        re.DOTALL matches all characters, including newlines.
    """
    if greedy:
        pattern = re.escape(substr1) + r'(.*)' + re.escape(substr2)
    else:
        pattern = re.escape(substr1) + r'(.*?)' + re.escape(substr2)
    positions = []

    for match in re.finditer(pattern, s, re.DOTALL):
        start = match.start() + len(substr1) # Exclude substr1
        end = match.end() - len(substr2) # Exclude substr2
        content = match.group(0)
        positions.append((start, end))

    return positions


def has_intersection(interval1, interval2):
    """
    Check if two intervals have an intersection.

    Args:
        interval1 (tuple): The first interval, represented as a tuple (start, end).
        interval2 (tuple): The second interval, represented as a tuple (start, end).

    Returns:
        bool: True if the intervals have an intersection, False otherwise.
    """
    return not (interval1[1] <= interval2[0] or interval1[0] >= interval2[1])

def num_lines(file):
    """
    Count the number of lines in a file.
    
    Args:
        file (str): The path to the file.
        
    Returns:
        int: The number of lines in the file.
    """
    if not os.path.exists(file):
        return 0

    with open(file, "r") as f:
        return len(f.readlines())

def read_lines(file):
    """
    Read all lines from a file.
    
    Args:
        file (str): The path to the file.
        
    Returns:
        List[str]: A list of lines in the file.
    """
    if not os.path.exists(file):
        return []

    with open(file, "r") as f:
        return f.readlines()
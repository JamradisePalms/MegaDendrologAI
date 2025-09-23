import json
from pathlib import Path
from typing import Union, Any, List, Dict

def parse_json_string(json_string: str) -> dict:
    """
    Parse a JSON-formatted string and return the resulting dictionary.

    Args:
        json_string (str): A string containing JSON data.

    Returns:
        dict: The parsed JSON data as a Python dictionary.
    """
    if json_string.startswith('```json'):
        json_string = json_string[7:]
    if json_string.endswith('```'):
        json_string = json_string[:-3]

    parsed_json = json.loads(json_string)
    return parsed_json


def write_json(
    obj_to_save: Union[Dict[str, Any], List[Dict[str, Any]], Any],
    path_where_to_save_json: Path
) -> None:
    """
    Save a dictionary or list of dictionaries as a JSON file.

    Args:
        obj_to_save (dict | list[dict] | Any): The dictionary or list of dictionaries to be saved as JSON.
        path_where_to_save_json (str, optional): The file path where the JSON will be saved.
            Defaults to 'LLM_responses/NewFeatures/features.json'.

    Returns:
        None
    """
    with open(path_where_to_save_json, 'w', encoding='Utf-8') as f:
        json.dump(obj_to_save, f, indent=4, ensure_ascii=False)
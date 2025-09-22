import json

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

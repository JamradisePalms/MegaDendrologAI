from pydantic import BaseModel
import json
from typing import Dict, Any, Type


class ResponseParser():
    def __init__(self, pydantic_model: Type[BaseModel]):
        self.dataclass = pydantic_model
    
    def parse(self, string_to_parse: str) -> Dict[str, Any]:
        left_boundary = -len(self.dataclass.model_fields) - 4 # We expect lines such as ```json and ``` in start and end of json lines and lines such as { and } (thats why - 4)
        lines = string_to_parse.splitlines()
        json_lines = lines[left_boundary:]

        if '```json' in lines[0]:
            json_lines = json_lines[1:-1]
        else:
            json_lines = json_lines[2:]
        json_string = '\n'.join(json_lines)
        print(json_string)
        result = json.loads(json_string)
        return result

import os
import requests
import base64
import mimetypes
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple
from configs.paths import PathConfig
PROMPT_FILEPATH = PathConfig.ML.Classification.PROMPT_FILEPATH

SOY_TOKEN = os.environ.get("SOY_TOKEN")
if not SOY_TOKEN:
    raise ValueError("Can`t find soy token")

class BaseClassifier(ABC):
    def __init__(self):
        pass
    
    @staticmethod
    def _encode_image(image_path: Path) -> Tuple[str, str]:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image'):
            raise ValueError(f"Couldn`t identify type of image: {image_path}")

        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        return encoded_string, mime_type

    @abstractmethod
    def single_request(self, image_path: Path) -> str:
        pass

class GptClassifier(BaseClassifier):
    def __init__(self, prompt_path: Path):
        self.prompt_path = PROMPT_FILEPATH

    def single_request(self, image_path: Path) -> str:
        url = "https://api.eliza.yandex.net/openai/v1/chat/completions"
        model_name = "gpt-5"
        with open(self.prompt_path, 'r', encoding='Utf-8') as f:
            message = f.read()

        user_content = [{"type": "text", "text": message}]
        
        if image_path:
            base64_image, mime_type = self._encode_image(image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
            })

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": user_content}]
        }
        headers = {
            "authorization": f"OAuth {SOY_TOKEN}",
            "content-type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers, verify=False)
        response.raise_for_status()
        return response.json()['response']['choices'][0]['message']['content']

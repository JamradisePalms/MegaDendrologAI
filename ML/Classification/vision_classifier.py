import os
import requests
import base64
import mimetypes
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, List, Union, Dict, Any, Iterable
from configs.paths import PathConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from ML.Classification.utils import parse_json_string
import urllib3
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)

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
        if not mime_type or not mime_type.startswith("image"):
            raise ValueError(f"Couldn`t identify type of image: {image_path}")

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        return encoded_string, mime_type

    @abstractmethod
    def _single_request(self, image_path: Path) -> str:
        pass


class GptClassifier(BaseClassifier):
    def __init__(self):
        self.prompt_path = PROMPT_FILEPATH

    def _single_request(self, image_path: Path) -> str:
        url = "https://api.eliza.yandex.net/openai/v1/chat/completions"
        model_name = "gpt-5"
        with open(self.prompt_path, "r", encoding="Utf-8") as f:
            message = f.read()

        user_content = [{"type": "text", "text": message}]

        if image_path:
            base64_image, mime_type = self._encode_image(image_path)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                }
            )

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": user_content}],
        }
        headers = {
            "authorization": f"OAuth {SOY_TOKEN}",
            "content-type": "application/json",
        }

        response = requests.post(url, json=payload, headers=headers, verify=False)
        response.raise_for_status()
        return response.json()["response"]["choices"][0]["message"]["content"]

    def run(
        self, images: Union[Iterable[Path], Path], max_workers: int = 5
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        if isinstance(images, Path):
            return parse_json_string(self._single_request(images))

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            api_requests = [
                executor.submit(self._single_request, image_path)
                for image_path in images
            ]
            api_requests_in_process = tqdm(as_completed(api_requests), total=len(api_requests))
            for response in api_requests_in_process:
                parsed_respose = parse_json_string(response.result())
                results.append(parsed_respose)

        return results

if __name__ == "__main__":
    classifier = GptClassifier()
    images = Path("ML/Classification/images").iterdir()
    response = classifier.run(images)

    print(response)

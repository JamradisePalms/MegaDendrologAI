from ML.Classification.vlm_lib.BaseClassifier import BaseClassifier
from ML.Classification.vlm_lib.Parser import ResponseParser
from ML.Classification.vlm_lib.Dataclasses import ClassificationTreeAnalysis
import requests
import os
from pydantic import BaseModel
from pathlib import Path
from typing import Type, Dict, Any
from configs.paths import PathConfig
import urllib3
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)

CLASSIFICATION_PROMPT = PathConfig.ML.Classification.CLASSIFICATION_PROMPT_FILEPATH
DEBUG_IMAGES = PathConfig.ML.Classification.PATH_TO_DEBUG_IMAGES

SOY_TOKEN = os.environ.get("SOY_TOKEN")
class YndxImageClassifier(BaseClassifier):
    pydantic_model: BaseModel

    def __init__(self, prompt_filepath, pydantic_model: Type[BaseModel]):
        super().__init__(prompt_filepath)
        self.pydantic_model = pydantic_model
        self.parser = ResponseParser(pydantic_model)

    def _single_request(self, image_path: Path) -> Dict[str, Any]:
        url = "https://api.eliza.yandex.net/internal/vlm/yandex_vlm_pro_soy/generative"
        model_name = "yandex_vlm_pro_soy"

        messages = []

        messages.append({"role": "user", "content": self.prompt})

        content = []
        content.append({"type": "text", "text": self.prompt})

        base64_image, mime_type = self._encode_image(image_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            }
        )

        messages.append({"role": "user", "content": content})

        payload = {
            "model": model_name,
            "messages": messages,
        }
        headers = {
            "authorization": f"OAuth {SOY_TOKEN}",
            "content-type": "application/json",
        }

        response = requests.post(url, json=payload, headers=headers, verify=False)
        response.raise_for_status()
        result_json = response.json()["response"]["vlm"][0]["Response"]
        parsed_answer = self.parser.parse(result_json)

        return self.pydantic_model.model_validate(parsed_answer)

if __name__ == '__main__':
    cl = YndxImageClassifier(CLASSIFICATION_PROMPT, ClassificationTreeAnalysis)
    response = cl.run(DEBUG_IMAGES.iterdir(), max_workers=6)
    
    print(response)
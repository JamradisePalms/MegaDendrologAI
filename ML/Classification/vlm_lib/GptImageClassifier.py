import os
import requests
from pathlib import Path
from configs.paths import PathConfig
from ML.Classification.vlm_lib.utils import parse_json_string
from ML.Classification.vlm_lib.BaseClassifier import BaseClassifier
from ML.Classification.vlm_lib.Dataclasses import TreeAnalysis
import urllib3
from urllib3.exceptions import InsecureRequestWarning

urllib3.disable_warnings(InsecureRequestWarning)
CLASSIFICATION_PATHS = PathConfig.ML.Classification

# prompt filepaths
CLASSIFICATION_PROMPT_FILEPATH = CLASSIFICATION_PATHS.CLASSIFICATION_PROMPT_FILEPATH
TREE_TYPE_PROMPT_FILEPATH = CLASSIFICATION_PATHS.TREE_TYPE_PROMPT_FILEPATH

# path to images to post to VLM
PATH_TO_TEST_IMAGES = CLASSIFICATION_PATHS.PATH_TO_TEST_IMAGES

# json filepaths where to save results
TREE_TYPE_DATASET_FILEPATH = CLASSIFICATION_PATHS.TREE_TYPE_DATASET_FILEPATH
CLASSIFICATION_DATASET_FILEPATH = CLASSIFICATION_PATHS.CLASSIFICATION_DATASET_FILEPATH


SOY_TOKEN = os.environ.get("SOY_TOKEN")
if not SOY_TOKEN:
    raise ValueError("Can`t find soy token")


class GptImageClassifier(BaseClassifier):
    def __init__(self, prompt_path: Path):
        super().__init__(prompt_path)

    def _single_request(self, image_path: Path) -> TreeAnalysis:
        url = "https://api.eliza.yandex.net/openai/v1/chat/completions"
        model_name = "gpt-5"

        user_content = [{"type": "text", "text": self.prompt}]

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
        return TreeAnalysis.model_validate(
            parse_json_string(
                response.json()["response"]["choices"][0]["message"]["content"]
            )
        )


if __name__ == "__main__":
    classifier = GptImageClassifier(CLASSIFICATION_PROMPT_FILEPATH)
    response = classifier.run(list(PATH_TO_TEST_IMAGES.iterdir()), max_workers=10)
    print(response)

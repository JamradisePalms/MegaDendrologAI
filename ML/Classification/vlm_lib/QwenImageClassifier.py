import json
from qwen_api import Qwen
from qwen_api.core.exceptions import QwenAPIError
from qwen_api.core.types.chat import ChatMessage, TextBlock, ImageBlock
from pathlib import Path
from configs.paths import PathConfig

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
PATH_TO_DEBUG_IMAGES = CLASSIFICATION_PATHS.PATH_TO_DEBUG_IMAGES

# json filepaths where to save results
TREE_TYPE_DATASET_FILEPATH = CLASSIFICATION_PATHS.TREE_TYPE_DATASET_FILEPATH
CLASSIFICATION_DATASET_FILEPATH = CLASSIFICATION_PATHS.CLASSIFICATION_DATASET_FILEPATH


class QwenImageClassifier(BaseClassifier):
    def __init__(self, prompt_path):
        super().__init__(prompt_path)

    def _single_request(self, image_path: Path) -> TreeAnalysis:
        client = Qwen()
        try:
            getdataImage = client.chat.upload_file(file_path=str(image_path))

            messages = [
                ChatMessage(
                    role="user",
                    web_search=False,
                    thinking=True,
                    blocks=[
                        TextBlock(block_type="text", text=self.prompt),
                        ImageBlock(
                            block_type="image",
                            url=getdataImage.file_url,
                            image_mimetype=getdataImage.image_mimetype,
                        ),
                    ],
                )
            ]

            response = client.chat.create(
                messages=messages,
                model="qwen3-vl-plus",
                stream=True,
            )
            string_response = []
            for chunk in response:
                delta = chunk.choices[0].delta
                if "extra" in delta and "web_search_info" in delta.extra:
                    print("\nSearch results:", delta.extra.web_search_info)
                    print()
                string_response.append(delta.content)
            # print(''.join(string_response))
            response_lines = "".join(string_response).splitlines()[-9:]
            if response_lines[0].startswith("```json"):
                response_lines = response_lines[1:-1]
            else:
                response_lines = response_lines[2:]

            result = json.loads("\n".join(response_lines))

            return TreeAnalysis.model_validate(result)
        except QwenAPIError as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    classifier = QwenImageClassifier(CLASSIFICATION_PROMPT_FILEPATH)
    response = classifier.run(PATH_TO_DEBUG_IMAGES.iterdir())
    print(response)
    print(len(response))

from dotenv import load_dotenv
load_dotenv()

import json
from typing import Iterable, Union, Type
from qwen_api import Qwen
from qwen_api.core.exceptions import QwenAPIError
from qwen_api.core.types.chat import ChatMessage, TextBlock, ImageBlock
from pathlib import Path
from configs.paths import PathConfig
from pydantic import BaseModel

from ML.Classification.vlm_lib.BaseClassifier import BaseClassifier
from ML.Classification.vlm_lib.Parser import ResponseParser
from ML.Classification.vlm_lib.Dataclasses import DetectionTreeAnalysis, ClassificationTreeAnalysis
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
    def __init__(self, prompt_path: Path, pydantic_model: Type[BaseModel]):
        super().__init__(prompt_path)
        self.pydantic_model = pydantic_model
        self.parser = ResponseParser(pydantic_model)

    def _single_request(self, image_path: Union[Path, Iterable[Path]]) -> BaseModel:
        client = Qwen()
        try:

            blocks = [TextBlock(block_type="text", text=self.prompt)]
            if isinstance(image_path, Iterable):
                for image in image_path:
                    getdataImage = client.chat.upload_file(file_path=str(image))
                    blocks.append(
                        ImageBlock(
                            block_type="image",
                            url=getdataImage.file_url,
                            image_mimetype=getdataImage.image_mimetype,
                        )
                    )
            else:
                getdataImage = client.chat.upload_file(file_path=str(image_path))
                blocks.append(
                    ImageBlock(
                        block_type="image",
                        url=getdataImage.file_url,
                        image_mimetype=getdataImage.image_mimetype,
                    )
                )
            messages = [
                ChatMessage(
                    role="user",
                    web_search=False,
                    thinking=True,
                    blocks=blocks,
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
            print(string_response)
            string_response = "".join(string_response)
            print(string_response)
            parsed_response = self.parser.parse(string_response)
            # return self.pydantic_model.model_validate(parsed_response)
            print(parsed_response)
            return parsed_response
        except QwenAPIError as e:
            print(f"Error: {str(e)}")


# qvq-72b-preview-0310
# qwen3-vl-plus
if __name__ == "__main__":
    classifier = QwenImageClassifier(
        CLASSIFICATION_PROMPT_FILEPATH, ClassificationTreeAnalysis
    )
    iterator = PATH_TO_DEBUG_IMAGES.iterdir()
    images_to_run = next(iterator)
    response = classifier.run(images_to_run)
    print(response)
    print(len(response))

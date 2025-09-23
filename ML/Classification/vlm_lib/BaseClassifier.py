import base64
import mimetypes
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, List, Union, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from ML.Classification.vlm_lib.Dataclasses import TreeAnalysis


class BaseClassifier(ABC):
    def __init__(self, prompt_path):
        self.prompt_path = prompt_path
        with open(self.prompt_path, "r", encoding="Utf-8") as f:
            message = f.read()
        self.prompt = message

    @staticmethod
    def _encode_image(image_path: Path) -> Tuple[str, str]:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith("image"):
            raise ValueError(f"Couldn`t identify type of image: {image_path}")

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        return encoded_string, mime_type

    @abstractmethod
    def _single_request(self, image_path: Path) -> TreeAnalysis:
        pass

    def run(
        self, images: Union[Iterable[Path], Path], max_workers: int = 5
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        if isinstance(images, Path):
            return self._single_request(images).model_dump()

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            api_requests = {
                executor.submit(self._single_request, image_path): image_path
                for image_path in images
            }
            api_requests_in_process = tqdm(
                as_completed(api_requests), total=len(api_requests)
            )
            for response in api_requests_in_process:
                image_path = api_requests[response]
                try:
                    result = response.result()
                    results.append(
                        {
                            **result.model_dump(),
                            'image': image_path
                        }
                    )
                except:
                    continue
        return results

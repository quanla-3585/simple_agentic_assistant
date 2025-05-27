from __future__ import annotations

import json
import re
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type

from pydantic import BaseModel


class LLMInput(BaseModel):
    user_prompt: str
    system_prompt: str = ''
    json_mode: bool = False
    response_model: Optional[Type[BaseModel]] = None
    file_content: bytes | None = None


class LLMBaseService(BaseModel):
    @property
    @abstractmethod
    def client(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def _inference_by_llm(self, message: LLMInput) -> Any:
        raise NotImplementedError()

    def postprocess(self, response_text: str) -> str:
        """Postprocess response text

        Args:
            response_text (str): response text from llm model

        Returns:
            str: postprocessed response text
        """
        response_text = response_text.replace('\n', '')
        response_text = response_text.strip()

        return response_text

    def json_parse(self, text: str) -> Dict:
        """Parse JSON object from text

        Args:
            text (str): output from llm model

        Returns:
            Dict: JSON object
        """
        # postprocess response text
        text = self.postprocess(text)

        # extract json object from text
        curly_brackets_match = re.search(r'\{.*\}', text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL)
        if curly_brackets_match:
            json_str = curly_brackets_match.group()
            try:
                json_object = json.loads(json_str, strict=False)
            except json.decoder.JSONDecodeError:
                json_object = {}
        else:
            json_object = {}

        return json_object
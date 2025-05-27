from .settings import AWSClaudeSetting

from .llm_service import LLMBaseService
from .llm_service import LLMInput
import asyncio
import json
import re
from typing import Union

import boto3
import instructor
from botocore.config import Config
from fastapi.encoders import jsonable_encoder
from instructor import Instructor
from pydantic import BaseModel
from logging import getLogger

logger = getLogger(__name__)


class AWSClaudeInput(LLMInput):
    pass


class AWSClaudeOutput(BaseModel):
    response: Union[str, dict, BaseModel]


class AWSClaudeService(LLMBaseService):
    settings: AWSClaudeSetting

    @property
    def inference_config(self):
        """Inference config for AWSClaude

        Returns:
            dict: inference config
        """
        return {
            'maxTokens': self.settings.max_tokens,
            'temperature': self.settings.temperature,
            'topP': self.settings.top_p,
        }

    @property
    def client(self) -> Instructor:
        """Create AWSClaude BedrockRuntime client

        Returns:
            boto3.client: AWSClaude BedrockRuntime client
        """
        try:
            config = Config(read_timeout=self.settings.read_timeout, connect_timeout=self.settings.connect_timeout)
            bedrock_client = boto3.client(
                service_name=self.settings.service_name,
                region_name=self.settings.region,
                aws_access_key_id=self.settings.access_key,
                aws_secret_access_key=self.settings.secret_access_key,
                config=config,
            )

            # Enable instructor patches for Bedrock client
            client = instructor.from_bedrock(bedrock_client)

            return client
        except Exception as e:
            logger.exception(
                f'Error occurred while creating AWS Sonnet client: {str(e)}', extra={},
            )
            raise e

    def _inference_by_llm(
        self,
        message: AWSClaudeInput,
    ) -> Union[str, dict, BaseModel]:
        """Inference by LLM model

        Args:
            message (AWSClaudeInput): AWSClaudeInput object

        Returns:
            Optional[Union[str, dict]]: response text
        """
        user_message = self._create_user_message(message.user_prompt)
        if message.system_prompt:
            system = self._create_system_message(message.system_prompt)

            response = self.client.chat.completions.create(
                modelId=self.settings.llm_model_id,
                messages=user_message,  # type: ignore
                system=system,
                response_model=message.response_model,  # type: ignore
            )
        else:
            response = self.client.chat.completions.create(
                modelId=self.settings.llm_model_id,
                messages=user_message,  # type: ignore
                response_model=message.response_model,  # type: ignore
            )

        if message.json_mode:
            return json.loads(response['output']['message']['content'][0]['text'], strict=False)
        if isinstance(response, BaseModel):
            return response
        return response['output']['message']['content'][0]['text']

    def process(self, inputs: AWSClaudeInput) -> AWSClaudeOutput:
        return asyncio.run(self.a_process(inputs))

    async def a_process(self, inputs: AWSClaudeInput) -> AWSClaudeOutput:
        """Call request to AWSClaude
        Args:
            inputs (AWSClaudeInput): AWSClaudeInput object

        Returns:
            AWSClaudeOutput: AWSClaudeOutput object
        """
        response = self._inference_by_llm(
            message=inputs,
        )
        if not response:
            logger.error(
                'Error occurred while processing request', response=response, extra=jsonable_encoder(inputs),
            )
            raise Exception('Error on processing request to LLM')
        return AWSClaudeOutput(response=response)

    def embedding(self, text: str) -> list[float]:
        """Call request to embedding model
        Args:
            text (str): text

        Returns:
            list[float]: embedding list
        """
        accept = 'application/json'
        content_type = 'application/json'

        response = self.client.invoke_model(
            body=json.dumps(
                {
                    'inputText': text,
                },
                ensure_ascii=False,
            ),
            modelId=self.settings.embedding_model_id,
            accept=accept,
            contentType=content_type,
        )

        response_body = json.loads(response.get('body').read())
        return response_body.get('embedding', [])

    def _create_user_message(self, user_prompt: str) -> list[dict]:
        """Create user message with given prompt

        Args:
            user_prompt (str): user prompt
        Returns:
            list[dict]: user message
        """
        return [
            {
                'role': 'user',
                'content': [
                    {
                        'text': user_prompt,
                    },
                ],
            },
        ]

    def _create_system_message(self, system_prompt: str) -> list[dict]:
        """Create system message with given prompt

        Args:
            system_prompt (str): system prompt
        Returns:
            list[dict]: system message
        """
        return [
            {
                'text': system_prompt,
            },
        ]

    async def pdf_extractor(self, inputs: AWSClaudeInput) -> str:
        """
        Call request to LLM model with PDF file content and user prompt

        Args:
            inputs (AWSClaudeInput): user prompt and PDF file content

        Returns:
            str: response text
        """
        user_message = self._create_user_message(inputs.user_prompt)
        user_message[0]['content'].append({
            'document': {
                'name': 'PDF Document',
                'format': 'pdf',
                'source': {'bytes': inputs.file_content},
            },
        })

        response = self.client.converse(
            modelId=self.settings.llm_model_id,
            messages=user_message,
            inferenceConfig=self.inference_config,
        )
        result = response['output']['message']['content'][0]['text']
        result_clean = re.sub(r'[\u3000]', ' ', result)
        return result_clean
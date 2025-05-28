from pydantic import BaseModel
from pydantic_settings import BaseSettings

class AWSClaudeSetting(BaseModel):
    service_name: str
    llm_model_id: str
    region: str
    access_key: str
    secret_access_key: str
    max_tokens: int
    temperature: float
    top_p: float
    read_timeout: int
    connect_timeout: int


class AzureOpenAISetting(BaseModel):
    api_key: str
    endpoint: str
    api_version: str
    deployment_name: str
    temperature: float
   
class Settings(BaseSettings):
    aws_claude: AWSClaudeSetting
    azure_openai: AzureOpenAISetting

    class Config:
        env_file = ".env"
        env_nested_delimiter = '__'
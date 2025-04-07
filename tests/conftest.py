import pytest
import os
import sys
from unittest.mock import patch, MagicMock

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

os.environ["OPENAI_API_KEY"] = "sk-mock-api-key-for-testing"

@pytest.fixture(autouse=True)
def disable_api_calls():
    with patch("openai.OpenAI"), patch("openai.AsyncOpenAI"):
        yield

@pytest.fixture(autouse=True)
def patch_prompt_types():
    mock_chat_openai = MagicMock()
    
    with patch("langchain_decorators.common.ChatOpenAI", return_value=mock_chat_openai):
        yield 
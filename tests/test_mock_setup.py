import pytest
import sys

def test_prompt_template_mock():
    """Test that PromptTemplate is correctly mocked."""
    from langchain.prompts import PromptTemplate
    
    template = PromptTemplate.from_template("This is a {test}")
    assert hasattr(template, 'template')
    assert template.template == "This is a {test}"
    
def test_langchain_decorators_import():
    """Test that importing langchain_decorators doesn't raise errors."""
    try:
        import langchain_decorators
        # No assertion needed - if it imports without error, test passes
    except Exception as e:
        pytest.fail(f"Failed to import langchain_decorators: {e}")

def test_langchain_decorators_still_works():
    """Test that langchain_decorators is not mocked and still works."""
    # Import should succeed
    import langchain_decorators
    
    # Test some functionality
    from langchain_decorators import llm_prompt
    
    # Verify the decorator exists and is callable
    assert callable(llm_prompt) 
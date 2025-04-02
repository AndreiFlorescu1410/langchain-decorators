import datetime
import logging
from textwrap import dedent
from typing import Callable, Dict, List, Type, TypeVar, Union, get_origin, get_args, Any, Generic, Optional
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseOutputParser, OutputParserException
import re
import json
import yaml
from .function_decorator import llm_function
from .pydantic_helpers import *

from pydantic import BaseModel, ValidationError, Field
from pydantic.json_schema import model_json_schema
from pydantic.fields import FieldInfo

class ErrorCodes:
    UNSPECIFIED = 0
    INVALID_FORMAT = 10
    INVALID_JSON = 15
    DATA_VALIDATION_ERROR = 20

class OutputParserExceptionWithOriginal(OutputParserException):
    """Exception raised when an output parser fails to parse the output of an LLM call."""    

    def __init__(self, message: str, original: str, original_prompt_needed_on_retry:bool=False, error_code:int=0) -> None:
        super().__init__(message)
        self.original = original
        self.observation=message
        self.error_code=error_code
        self.original_prompt_needed_on_retry=original_prompt_needed_on_retry

    def __str__(self) -> str:
        return f"{super().__str__()}\nOriginal output:\n{self.original}"


class ListOutputParser(BaseOutputParser):
    """Class to parse the output of an LLM call in a bullet/numbered list format to a list."""

    @property
    def _type(self) -> str:
        return "list"

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""

        pattern = r"^[ \t]*(?:[\-\*\+]|\d+\.)[ \t]+(.+)$"
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if not matches and text:
            logging.warning(
                f"{self.__class__.__name__} : LLM returned {text} but we could not parse it into a list")
        return matches

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        return "Return result a s bulleted list."

class BooleanOutputParser(BaseOutputParser):
    """Class to parse the output of an LLM call to a boolean."""
    pattern:str
    
    @property
    def _type(self) -> str:
        return "boolean"
    
    def __init__(self, pattern: str = r"((Yes)|(No))([,|.|!]|$)") -> None:
        super().__init__(pattern=pattern)

    def parse(self, text: str) -> bool:
        """Parse the output of an LLM call."""

        
        
        match = re.search(self.pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        
        if not match:
            raise OutputParserExceptionWithOriginal(message=self.get_format_instructions(),original=text, original_prompt_needed_on_retry=True, error_code=ErrorCodes.INVALID_FORMAT)
        else:
            return match.group(1).lower() == "yes"
        

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        return "Reply only Yes or No.\nUse this format: Final decision: Yes/No"

class JsonOutputParser(BaseOutputParser):
    """Class to parse the output of an LLM call to a Json."""

    @property
    def _type(self) -> str:
        return "json"

    def find_json_block(self,text, raise_if_not_found=True):

        start_code_block = list(re.finditer(r"(\n|^)```((json)|\n)", text))
        if start_code_block:
            i_start = start_code_block[0].span()[1]
            end_code_block = list(re.finditer(r"\n```($|\n)", text[i_start:]))
            if end_code_block:
                i_end = end_code_block[0].span()[0]
                text = text[i_start : i_start + i_end]

        match = re.search(r"[\{|\[].*[\}|\]]", text.strip(),
                              re.MULTILINE | re.IGNORECASE | re.DOTALL)
        if not match and raise_if_not_found:
            raise OutputParserExceptionWithOriginal(message="No JSON found in the response", original_output=text, error_code=ErrorCodes.INVALID_JSON)
        return match

    def replace_json_block(self, text: str, replace_func:Callable[[dict],str]) -> str:
        try:

            match = self.find_json_block(text)
            json_str = match.group()
            i_start = match.start()
            _i_start = ("\n"+text).rfind("\n```", 0, i_start)
            i_end = match.end()
            _i_end = text.find("\n```\n", i_end)
            i_start=_i_start if _i_start>=0 else i_start
            i_end=_i_end+5 if _i_end>=0 else i_end

            json_dict = json.loads(json_str, strict=False)
            replacement = replace_func(json_dict)
            return (text[:i_start] + replacement + text[i_end:]).strip()

        except (json.JSONDecodeError) as e:

            msg = f"Invalid JSON\n {text}\nGot: {e}"
            raise OutputParserExceptionWithOriginal(msg, text, error_code=ErrorCodes.INVALID_JSON)

    def parse(self, text: str) -> dict:
        try:
            # Greedy search for 1st json candidate.
            match = self.find_json_block(text)
            json_str = match.group()
            try:
                json_dict = json.loads(json_str, strict=False)
            except json.JSONDecodeError as e:
                try:
                    from json_repair import repair_json
                    repair_json = repair_json(json_str)
                    json_dict = json.loads(repair_json, strict=False)
                    return json_dict
                except ImportError:
                    logging.warning("We might have been able to fix this output using json_repair. You can try json autorepair by installing json_repair package (`pip install json_repair`)")
                    pass
                raise e
            return json_dict

        except (json.JSONDecodeError) as e:

            msg = f"Invalid JSON\n {text}\nGot: {e}"
            raise OutputParserExceptionWithOriginal(msg, text, error_code=ErrorCodes.INVALID_JSON)

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        return "Return result as a valid JSON"


T = TypeVar("T", bound=BaseModel)


class PydanticOutputParser(BaseOutputParser[T]):
    """Class to parse the output of an LLM call to a pydantic object."""
    model: Type[T]
    as_list: bool = False
    instructions_as_json_example: bool = True

    def __init__(self, model: Type[T], instructions_as_json_example: bool = True, as_list: bool = False):
        super().__init__(model=model, instructions_as_json_example=instructions_as_json_example,as_list=as_list)

    @property
    def _type(self) -> str:
        return "pydantic"
    
    def parse(self, text: str) -> T:
        try:
            # Greedy search for 1st json candidate.
            regex_pattern = r"\[.*\]" if self.as_list else r"\{.*\}"
            match = re.search(regex_pattern, text.strip(),re.MULTILINE | re.IGNORECASE | re.DOTALL)
            json_str = ""
            if match:
                json_str = match.group()
            json_dict = json.loads(json_str, strict=False)
            if self.as_list:
                return [self.model.model_validate(item) for item in json_dict]
            else:        
                return self.model.model_validate(json_dict)

        except (json.JSONDecodeError) as e:
            msg = f"Invalid JSON\n {text}\nGot: {e}"
            raise OutputParserExceptionWithOriginal(msg, text, error_code=ErrorCodes.INVALID_JSON)

        except ValidationError as e:
            msg = f"Failed to parse {self.model.__name__}:\n{humanize_pydantic_validation_error(e)}"
            raise OutputParserExceptionWithOriginal(msg, text, error_code=ErrorCodes.DATA_VALIDATION_ERROR)

    def get_json_example_description(self, model:Type[BaseModel]=None, indentation_level=0):
        """Get a description of the model as a JSON example."""
        model = model or self.model
        schema = model_json_schema(model)
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        lines = []
        indent = "  " * indentation_level
        
        for field_name, field_schema in properties.items():
            field_type = field_schema.get("type", "any")
            field_description = field_schema.get("description", "")
            is_required = field_name in required
            
            if field_type == "object" and "properties" in field_schema:
                # Nested object
                lines.append(f"{indent}{field_name}: {{  # {field_description}")
                lines.append(self.get_json_example_description(field_schema, indentation_level + 1))
                lines.append(f"{indent}}}")
            elif field_type == "array" and "items" in field_schema:
                # Array type
                items_schema = field_schema["items"]
                if items_schema.get("type") == "object":
                    lines.append(f"{indent}{field_name}: [  # {field_description}")
                    lines.append(self.get_json_example_description(items_schema, indentation_level + 1))
                    lines.append(f"{indent}]")
                else:
                    example = f"[{items_schema.get('type', 'any')}]"
                    lines.append(f"{indent}{field_name}: {example}  # {field_description}")
            else:
                # Simple type
                example = field_schema.get("example", field_type)
                suffix = "" if is_required else " (optional)"
                lines.append(f"{indent}{field_name}: {example}  # {field_description}{suffix}")
        
        return "\n".join(lines)

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        if self.instructions_as_json_example:
            schema = model_json_schema(self.model)
            example = self.get_json_example_description()
            return dedent(f"""Return a JSON object that matches the following schema:
            {example}
            """)
        else:
            schema = model_json_schema(self.model)
            return dedent(f"""Return a JSON object that matches the following schema:
            {json.dumps(schema, indent=2)}
            """)

class OpenAIFunctionsPydanticOutputParser(BaseOutputParser[T]):
    model: Type[T]

    @property
    def _type(self) -> str:
        return "pydantic"

    def __init__(self, model: Type[T]):
        super().__init__(model=model)

    def parse(self, function_call_arguments:dict ) -> T:
        try:
            return self.model.model_validate(function_call_arguments)
        except ValidationError as e:
            msg = f"Failed to parse {self.model.__name__}:\n{humanize_pydantic_validation_error(e)}"
            raise OutputParserExceptionWithOriginal(msg, str(function_call_arguments), error_code=ErrorCodes.DATA_VALIDATION_ERROR)

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        return ""

    def build_llm_function(self):
        @llm_function(arguments_schema=self.model)
        def generate_response( **kwargs) -> T:
            """Generate response"""
            return kwargs

        return generate_response


class CheckListParser(ListOutputParser):
    """Parses list a a dictionary... assume this format:
        - KeyParma1: Value1
        - KeyPara2: Value2
        ...
    """

    def __init__(self, model: Type[T] = None):
        self.model = model

    @property
    def _type(self) -> str:
        return "checklist"

    def get_instructions_for_model(self, model: Type[T]) -> str:
        fields_bullets = []
        for field in model.__fields__.values():
            description = [field.field_info.description]
            if field.field_info.extra.get("one_of"):
                description += "one of these values: [ "
                description += " | ".join(field.field_info.extra.get("one_of"))
                description += " ]"
            if field.field_info.extra.get("example"):
                description += f"e.g. {field.field_info.extra.get('example')}"
            if description:
                description = " ".join(description)
            else:
                description = "?"
            fields_bullets.append(f"- {field.name}: {description}")

    def parse(self, text: str) -> Union[dict, T]:
        """Parse the output of an LLM call."""

        pattern = r"^[ \t]*(?:[\-\*\+]|\d+\.)[ \t]+(.+)$"
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        result = {}
        if not matches:
            raise OutputParserExceptionWithOriginal(message="No matches found", original_output=text, error_code=ErrorCodes.INVALID_FORMAT)
        for match in matches:
            key, value = match.split(":", 1)
            result[key.strip()] = value.strip()

        return matches

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        res = "Return result a s bulleted list in this format:\n"
        if self.model:
            res += self.get_instructions_for_model(self.model)
        else:
            res += "\n- Key1: Value1\n- Key2: Value2\n- ..."


class MarkdownStructureParser(ListOutputParser):
    model: Type[T] = None
    level: int = 1
    sections_parsers: Dict[str, Union[BaseOutputParser, dict]] = None

    def __init__(self,  model: Type[T] = None, sections_parsers: Dict[str, Union[dict, BaseOutputParser]] = None, level=1):

        super().__init__(model=model, sections_parsers=sections_parsers, level=level)
        if model:
            for field, field_info in model.__fields__.items():
                if sections_parsers and field in self.sections_parsers:
                    # if section parser was already provided, skip
                    if not type(self.sections_parsers.get(field)) == dict:
                        continue
                field_type = get_field_type(field_info)
                if get_field_type(field_info) == list:
                    item_type = get_field_item_type(field_info)
                    if item_type == str or item_type is None:
                        self.sections_parsers[field] = ListOutputParser()
                    else:
                        raise ValueError(
                            f"Unsupported item type {item_type} for property {model}.{field}. Only list of strings is supported.")
                elif field_type == dict:
                    self.sections_parsers[field] = CheckListParser()
                elif field_type and issubclass(field_type, BaseModel):

                    all_sub_str = all(True for sub_field_info in field_type.__fields__.values(
                    ) if get_field_type(sub_field_info) == str)

                    if all_sub_str:

                        self.sections_parsers[field] = MarkdownStructureParser(
                                model=field_type, sections_parsers=sections_parsers.get(field), level=level+1
                            )
                    else:
                        self.sections_parsers[field] = PydanticOutputParser(
                                model=field_type
                            )

                elif field_type == str:

                    self.sections_parsers[field] = None
                else:
                    raise ValueError(
                        f"Unsupported type {field_type} for property {field}.")
        elif sections_parsers:
            for property, property_parser in sections_parsers.items():
                if type(property_parser) == dict:
                    sections_parsers[property] = MarkdownStructureParser(
                        model=None, sections_parsers=property_parser, level=level+1)
                elif type(property_parser) == str:
                    sections_parsers[property] = None
                elif isinstance(property_parser, BaseOutputParser):
                    continue
                else:
                    raise ValueError(
                        f"Unsupported type {model.__fields__[property].annotation} for property {property}. Use a dict or a pydantic model.")
        else:
            self.sections_parsers = {}

    @property
    def _type(self) -> str:
        return "checklist"

    def get_instructions_for_sections(self,  model: Type[T] = None, sections_parsers: Dict[str, BaseOutputParser] = None) -> str:
        section_instructions = []
        if model:
            for field, field_info in model.__fields__.items():
                name: str = field_info.field_info.title or field
                section_instructions.append(self.level*"#" + f" {name}")
                if sections_parsers and sections_parsers.get(field):
                    section_instructions.append(
                        sections_parsers.get(field).get_format_instructions())
                    continue
                else:

                    description = _get_str_field_description(field_info)
                    section_instructions.append(description)
        else:
            for section, parser in sections_parsers.items():
                section_instructions.append(self.level*"#" + f" {section}")
                if isinstance(parser, BaseOutputParser):
                    section_instructions.append(
                        parser.get_format_instructions())
                else:
                    section_instructions.append("?")

        return "\n\n".join(section_instructions)

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""

        sections_separators = list(re.finditer(
            r"^#+[ |\t]+(.*)$", text, flags=re.MULTILINE))
        res = {}
        for i, section_separator_match in enumerate(sections_separators):

            section_name = section_separator_match.group(1)
            if self.model:
                section_name = next((field for field, field_info in self.model.__fields__.items() if field_info.field_info.title ==
                                    section_name or field.lower() == section_name.lower() or field_info.alias == section_name), section_name)
            if i < len(sections_separators)-1:
                section_content = text[section_separator_match.end(
                ):sections_separators[i+1].start()]
            else:
                section_content = text[section_separator_match.end():]

            parsed_content = None
            if self.sections_parsers and self.sections_parsers.get(section_name, None) or self.sections_parsers.get(section_separator_match.group(1)):
                parser = self.sections_parsers.get(
                    section_name, None) or self.sections_parsers.get(section_separator_match.group(1))
                if isinstance(parser, BaseOutputParser):
                    parsed_content = parser.parse(section_content)
            if not parsed_content:
                parsed_content = section_content.strip()

            res[section_name] = parsed_content

        if self.model:
            try:
                return self.model(**res)
            except ValidationError as e:
                try:
                    res_aligned = align_fields_with_model(res, self.model)
                    return self.model.parse_obj(res_aligned)
                except ValidationError as e:
                    err_msg =humanize_pydantic_validation_error(e)
                    raise OutputParserExceptionWithOriginal(f"Data are not in correct format: {text}\nGot: {err_msg}",text, error_code=ErrorCodes.DATA_VALIDATION_ERROR) 
        else:
            return res

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        res = "Return result as a markdown in this format:\n"
        if self.model or self.sections_parsers:
            res += self.get_instructions_for_sections(
                self.model, self.sections_parsers)

        else:
            res += "# Section 1\n\ndescription\n\n#Section 2\n\ndescription\n\n..."
        return res


def _get_str_field_description(field_info: FieldInfo, ignore_nullable: bool = False, default="?") -> str:
    """Get a string description of a field for LLM prompts."""
    field_type = get_field_type(field_info)
    is_nullable = is_field_nullable(field_info) if not ignore_nullable else False
    item_type = get_field_item_type(field_info)
    
    type_str = str(field_type.__name__ if hasattr(field_type, '__name__') else field_type)
    if item_type:
        item_type_str = item_type.__name__ if hasattr(item_type, '__name__') else str(item_type)
        type_str = f"{type_str}[{item_type_str}]"
    
    description = field_info.description or default
    nullable_str = " (optional)" if is_nullable else ""
    
    return f"{type_str}{nullable_str}: {description}"

def describe_field_schema(field_schema:dict):
    if "type" in field_schema:
        res = field_schema.pop("type")
        return res + ", " + ", ".join([f"{k}:{v}" for k,v in field_schema.items()])
    else:
        return ""

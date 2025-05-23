import inspect
import re
from enum import Enum
from functools import wraps
from string import Formatter
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel
from pydantic.json_schema import model_json_schema
from pydantic.fields import FieldInfo

from .common import (
    get_arguments_as_pydantic_fields,
    get_function_docs,
    get_function_full_name,
)
from .pydantic_helpers import sanitize_pydantic_schema


class DocstringsFormat(Enum):
    AUTO = "auto"
    GOOGLE = "google"
    SPHINX = "sphinx"
    NUMPY = "numpy"


def get_function_schema(func, schema_template_args=None):
    if callable(func) and hasattr(func, "get_function_schema"):
        return func.get_function_schema(func, schema_template_args)
    else:
        raise ValueError(
            f"Invalid item value in functions. Unable to retrieve schema from function {f}"
        )


def is_dynamic_llm_func(func):
    if callable(func) and hasattr(func, "get_function_schema"):
        return getattr(func, "is_dynamic", False)


def get_dynamic_function_template_args(func: Callable) -> Tuple[List[str], List[str]]:
    """
    returns tuple of (required_args:List[str], optional_args:List[str])
    """
    func_docs = get_function_docs(func)
    if not func_docs:
        raise ValueError(
            f"LLM Function {get_function_full_name(func)} has no docstring"
        )

    return get_template_args(func_docs)


class DotDict(dict):

    def __init__(self, dictionary: dict):
        super().__init__(dictionary)

    def __getattr__(self, name):
        if name in self:

            val = self.get(name)
            if isinstance(val, dict):
                return DotDict(val)
            else:
                return val
        else:
            return None


def llm_function(
    function_name: str = None,
    validate_docstrings: bool = True,
    docstring_format: str = "auto",
    arguments_schema: Union[Type[BaseModel], Type[List[Any]]] = None,
    dynamic_schema: bool = False,
    func_description: str = None,
    **kwargs,
):
    """
    Decorator for functions that take a language model as first argument.


    Args:
        - function_name: the name of the function for LLM. If not provided, the name of the function will be used

        - argument_schema
            - `auto` (default): the schema is automatically inferred from the function signature. If docstrings are provided, they will be used to enhance function description
            - `pydantic` -  expects a pydantic model as first and only argument ('self' being ignored) - allows for controllable schema
            - `docstring` - parses the schema ONLY from the docstring

            or define pydantic model:
            ```
            @llm_function(argument_schema=MyFunctionPydanticArgsSchema)
            def my_function(self, question:str)->bool:
                ...
            ```

            or define a simple dict with descriptions:
            ```
            @llm_function(argument_schema={"question": "Question to ask"})
            def my_function(self, question:str)->bool:
                ...
            ```

        - validate_docstrings: if True, the docstrings will be parsed and validated against function. If parsing or validation fails, an error will be raised

        - docstring_format: the format of the docstring
            -  `auto` (default): the format is automatically inferred from the docstring
            -  `google`: the docstring is parsed as markdown (see [Google docstring format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html))
            -  `numpy`: the docstring is parsed as markdown (see [Numpy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html))
            -  `sphinx`: the docstring is parsed as sphinx format (see [Sphinx docstring format](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html))

    Examples:

        Google style docstrings:
        ```python
        @llm_function
        def function_with_google_docstrings(self, question:str)->bool:
            \"\"\"
            This is a function with google docstrings
            Args:  # but accepts also Parameters, Arguments
                question (str): Question to ask
            \"\"\"
        ```

        ```python
        @llm_function
        def function_with_numpy_docstrings(self, question:str)->bool:
            \"\"\"
            This is a function with google docstrings
            Parameters # but accepts also Args, Arguments
            ----------
                question :int
                    Question to ask

            \"\"\"
        ```

        ```python
        @llm_function
        def function_with_sphinx_docstrings(self, question:str)->bool:
            \"\"\"
            This is a function with google docstrings
            :param question: Question to ask

            \"\"\"
        ```

    Note:
        The fact whether the parameter is optional is inferred from the function signature. You can also document it, but it will be just part of the description. If typing annotations are used (i.e. Optional[str]), this will be stripped to native type.

    """

    if callable(function_name):
        # this is the case when the decorator is called without arguments
        # we initialize params with default values
        func = function_name
        function_name = None
    else:
        func = None

    def decorator(func):

        if dynamic_schema:

            def get_function_schema(_func, schema_template_args):
                return build_func_schema(
                    _func,
                    function_name=function_name,
                    format=docstring_format,
                    validate_docstrings=validate_docstrings,
                    arguments_schema=arguments_schema,
                    func_description=func_description,
                    schema_template_parameters=schema_template_args,
                )

        else:
            func_schema = build_func_schema(
                func,
                function_name=function_name,
                format=docstring_format,
                validate_docstrings=validate_docstrings,
                arguments_schema=arguments_schema,
                func_description=func_description,
                schema_template_parameters=None,
            )

            def get_function_schema(_func, schema_template_args=None):
                return func_schema

        is_async = inspect.iscoroutinefunction(func)

        if not is_async:

            @wraps(func)
            def func_wrapper(*args, **kwargs):
                return func(*args, **kwargs)

        else:

            @wraps(func)
            async def func_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

        func_wrapper.get_function_schema = get_function_schema
        func_wrapper.is_dynamic = dynamic_schema
        func_wrapper.function_name = function_name or func.__name__

        return func_wrapper

    if func:
        return decorator(func)
    else:
        return decorator


class LllFunctionWithModifiedSchema:

    def __init__(self, func, modified_schema: dict):
        self.func = func
        self._function_schema = modified_schema

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def get_function_schema(self):
        return self._function_schema


def build_func_schema(
    func: Callable,
    function_name: str = None,
    format: Union[DocstringsFormat, str] = "auto",
    validate_docstrings: bool = True,
    arguments_schema: Union[Type[BaseModel], Type[List[Any]], None] = None,
    func_description: str = None,
    schema_template_parameters: Dict[str, Any] = None,
):
    """
    Build function schema for LLM function.
    """
    if isinstance(format, str):
        format = DocstringsFormat(format)

    if not function_name:
        function_name = func.__name__

    if not func_description:
        func_docs = get_function_docs(func)
        if func_docs:
            func_description = parse_function_description_from_docstrings(func_docs)

    if arguments_schema and issubclass(arguments_schema, BaseModel):
        # Use the Pydantic model's schema directly
        schema = model_json_schema(arguments_schema)
        schema["name"] = function_name
        if func_description:
            schema["description"] = func_description
        return sanitize_pydantic_schema(schema)

    # Get fields from function signature
    fields = get_arguments_as_pydantic_fields(func)
    
    # Create a dynamic model with these fields
    model = type(
        f"{function_name}Schema",
        (BaseModel,),
        {
            "__annotations__": {
                name: field.annotation for name, field in fields.items()
            },
            **{
                name: (field.default if field.default_factory is None else field.default_factory())
                for name, field in fields.items()
                if field.default is not None or field.default_factory is not None
            },
        },
    )

    # Generate schema
    schema = model_json_schema(model)
    schema["name"] = function_name
    if func_description:
        schema["description"] = func_description

    # Add docstring descriptions to properties if available
    if validate_docstrings:
        func_docs = get_function_docs(func)
        if func_docs:
            param_descriptions = find_and_parse_params_from_docstrings(func_docs, format)
            for param_name, description in param_descriptions.items():
                if param_name in schema.get("properties", {}):
                    schema["properties"][param_name]["description"] = description

    return sanitize_pydantic_schema(schema)


def format_str_extra(template: str, **kwargs):
    optional_blocks_regex = list(
        re.finditer(
            r"\{\?(?P<optional_partial>.+?)(?=\?\})\?\}",
            template,
            re.MULTILINE | re.DOTALL,
        )
    )
    for optional_block in optional_blocks_regex:
        optional_partial = optional_block.group("optional_partial")
        partial_input_variables = {
            v for _, v, _, _ in Formatter().parse(optional_partial) if v is not None
        }

        if not partial_input_variables:
            raise ValueError(
                f"Optional partial {optional_partial} does not contain any optional variables. Didn't you forget to wrap your parameter in {{}}?"
            )

        if not all(kwargs.get(v) for v in partial_input_variables):
            # all variables are provided, we can remove the optional block
            template = template.replace(optional_block.group(0), "")
        else:
            optional_partial = optional_block.group("optional_partial")
            template = template.replace(optional_block.group(0), optional_partial)

    _format_args = {
        k: v if not isinstance(v, dict) else DotDict(v) for k, v in kwargs.items()
    }
    return Formatter().format(template, **_format_args)


def get_template_args(template: str) -> Tuple[List[str], List[str]]:
    """returns tuple of (required_args:List[str], optional_args:List[str])"""
    optional_args = set()
    optional_blocks_regex = list(
        re.finditer(
            r"\{\?(?P<optional_partial>.+?)(?=\?\})\?\}",
            template,
            re.MULTILINE | re.DOTALL,
        )
    )
    for optional_block in optional_blocks_regex:
        optional_partial = optional_block.group("optional_partial")
        partial_input_variables = {
            v for _, v, _, _ in Formatter().parse(optional_partial) if v is not None
        }
        optional_args.update(partial_input_variables)
        if not partial_input_variables:
            raise ValueError(
                f"Optional partial {optional_partial} does not contain any optional variables. Didn't you forget to wrap your parameter in {{}}?"
            )

        # replace optional block with underscore so it wont be parsed for required args
        template = template.replace(optional_block.group(0), "_")

    required_args = {v for _, v, _, _ in Formatter().parse(template) if v is not None}
    return required_args, optional_args


def parse_function_description_from_docstrings(docstring: str) -> str:
    # we will return first text until first empty line

    lines = docstring.splitlines()
    description = []
    for line in lines:
        line = line.strip()
        if line:
            description.append(line)
        elif description:
            # if we have already some description, we stop at first empty line ... else continue
            break
    return "\n".join(description)


def find_and_parse_params_from_docstrings(
    docstring: str, format: DocstringsFormat
) -> str:
    """
    Find Args section in docstring.
    """

    args_section = None
    args_section_start = 0
    args_section_end = None

    if format == DocstringsFormat.AUTO or format == DocstringsFormat.GOOGLE:
        # auto here means more more free format than google
        args_section_start_regex_pattern = (
            r"(^|\n)(Args|Arguments|Parameters)\s*:?\s*\n"
        )
        args_section_end_regex_pattern = r"(^|\n)([A-Z][a-z]+)\s*:?\s*\n"
        if format == DocstringsFormat.GOOGLE:
            param_start_parser_regex = r"(^|\n)\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*(\((?P<type>[^\)]*)\))?\s*:\s*(?=[^\n]+)"
        else:
            param_start_parser_regex = r"(^|\n)\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*(\((?P<type>[^\)]*)\))?\s*(-|:)\s*(?=[^\n]+)"
    elif format == DocstringsFormat.NUMPY:
        args_section_start_regex_pattern = (
            r"(^|\n)(Args|Arguments|Parameters)\s*\n\s*---+\s*\n"
        )
        args_section_end_regex_pattern = r"(^|\n)([A-Z][a-z]+)\s*\n\s*---+\s*\n"
        param_start_parser_regex = r"(^|\n)\s*(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)(\s*:\s*(?P<type>[^\)]*)\s*)?\n\s+(?=[^\n]+)"
    elif format == DocstringsFormat.SPHINX:
        args_section_start_regex_pattern = None  # we will look for :param everywhere
        args_section_end_regex_pattern = r"(\n)\s*:[a-z]"
        param_start_parser_regex = r"(^|\n)\s*:param\s+(?P<type>[^\)]*)\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(?=[^\n]+)"

    if args_section_start_regex_pattern:
        match = re.search(args_section_start_regex_pattern, docstring)
        if match:
            args_section_start = match.end()
            if args_section_end_regex_pattern:
                match = re.search(
                    args_section_end_regex_pattern, docstring[args_section_start:]
                )
                if match:
                    args_section_end = match.start() + args_section_start
            if not args_section_end:
                args_section_end = len(docstring)
            args_section = docstring[args_section_start:args_section_end]
        else:
            args_section = None
    else:
        args_section = docstring

    params = {}
    if args_section:
        last_param = None
        last_param_end = None
        for param_start_match in re.finditer(param_start_parser_regex, args_section):
            if last_param_end is not None:
                last_param["description"] = args_section[
                    last_param_end : param_start_match.start()
                ].strip()

            param_name = param_start_match.group("name")
            param_type = param_start_match.group("type")
            last_param = {"type": param_type or "", "description": None}
            last_param_end = param_start_match.end()
            params[param_name] = last_param

        if last_param_end is not None:
            section_end = None
            if (
                args_section_start_regex_pattern is None
                and args_section_end_regex_pattern
            ):
                # this is handling SPHINX, we didnt parse the start so we cant parse the end until all the params are consumed... now we can parse the end after the last param
                section_end_match = re.search(
                    args_section_end_regex_pattern, docstring[last_param_end:]
                )
                if section_end_match:
                    section_end = last_param_end + section_end_match.start()
            if not section_end:
                section_end = len(docstring)
            last_param["description"] = args_section[last_param_end:section_end].strip()

    if not params and format == DocstringsFormat.AUTO:
        # try other options
        options = [DocstringsFormat.NUMPY, DocstringsFormat.SPHINX]
        for option in options:
            result = find_and_parse_params_from_docstrings(docstring, option)
            if result:
                return result
    else:
        return params


def parse_enum_from_docstring_param(type: str, description) -> str:
    enum_pattern = r"\[\s*(?P<value>[\"|'][\w|_|-| ]+[\"|']\s*(\||\]))+"
    enum_part_match = re.search(enum_pattern, type + " / " + description)
    if not enum_part_match:
        return None
    enum_strings = enum_part_match.group(0).strip("[]").split("|")
    return [enum.strip(" \"'") for enum in enum_strings]

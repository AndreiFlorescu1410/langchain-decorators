import logging
import re
import inspect
from abc import ABC, abstractmethod
from string import Formatter

from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union

from langchain.prompts import StringPromptTemplate, PromptTemplate
from langchain.prompts.chat import  MessagesPlaceholder, ChatMessagePromptTemplate, ChatPromptTemplate, ChatPromptValue
from langchain.schema import PromptValue, BaseOutputParser, BaseMemory, BaseChatMessageHistory

from promptwatch import register_prompt_template
from .schema import OutputWithFunctionCall
from .common import LogColors, PromptTypeSettings, get_func_return_type, get_function_docs, get_function_full_name, print_log
from .output_parsers import *

from pydantic import BaseModel

class BaseTemplateBuilder(ABC):

    @abstractmethod
    def build_template(self, template_parts:List[Tuple[str,str]],kwargs:Dict[str,Any])->PromptTemplate:
        """ Function that builds a prompt template from a template string and the prompt block name (which is the the part of ```<prompt:$prompt_block_name> in the decorated function docstring)

        Args:
            template_parts (List[Tuple[str,str]]): list of prompt parts List[(prompt_block_name, template_string)]
            kwargs (Dict[str,Any]): all arguments passed to the decorated function

        Returns:
            PromptTemplate: ChatPromptTemplate or StringPromptTemplate
        """
        pass

class OpenAITemplateBuilder:

    def build_template(self, template_parts:List[Tuple[str,str]],kwargs:Dict[str,Any])->PromptTemplate:
        if len(template_parts)==1 and not template_parts[0][1]:
            template_string=template_parts[0][0]
            return PromptTemplate.from_template(template_string)
        else:
            message_templates=[]
            for template_string, prompt_block_name in template_parts:
                template_string=template_string.strip()
                content_template= PromptTemplate.from_template(template_string)
                if prompt_block_name=="placeholder":
                    message_templates.append(MessagesPlaceholder(variable_name=template_string.strip(" {}")))
                elif prompt_block_name:
                    
                    if "[" in prompt_block_name and  prompt_block_name[-1]=="]":
                        i = prompt_block_name.find("[")
                        name = prompt_block_name[i+1:-1]
                        role=prompt_block_name[:i]
                    else:
                        name=None
                        role=prompt_block_name

                    if name:
                        additional_kwargs={"name":name}
                    elif role == "function":
                        raise Exception(f"Invalid function prompt block. function_name {name} is not set. Use this format: <prompt:function[function_name]>")
                    else:
                        additional_kwargs={}
                    
                    message_templates.append(ChatMessagePromptTemplate(role=role,prompt=content_template,additional_kwargs=additional_kwargs))
            return ChatPromptTemplate(messages=message_templates)
            
            



def parse_prompts_from_docs(docs:str):
    prompts = []
    for i, prompt_block in enumerate(re.finditer(r"```[^\S\n]*<prompt(?P<role>:[\w| |\[|\]]+)?>\n(?P<prompt>.*?)\n```[ |\t\n]*", docs, re.MULTILINE | re.DOTALL)):
        role = prompt_block.group("role")
        prompt = prompt_block.group("prompt")
        # remove \ escape before ```
        prompt = re.sub(r"((?<=\s)\\(?=```))|^\\(?=```)", "",prompt, flags=re.MULTILINE )
        prompt.strip()
        if not role:
            if i>1:
                raise ValueError("Only one prompt can be defined in code block. If you intend to define messages, you need to specify a role.\nExample:\n```<prompt:role>\nFoo {bar}\n```")
            else:
                prompts.append(prompt)
        else:
            prompts.append((role[1:], prompt))
    if not prompts:
        # the whole document is a prompt
        prompts.append(docs.strip())

    return prompts


class PromptTemplateDraft(BaseModel):
    role:str=None
    input_variables:List[str]
    template:str
    partial_variables_builders:Optional[Dict[str, Callable[[dict], str]]] = None

    def finalize_template(self, input_variable_values:dict)->Union[MessagesPlaceholder, ChatMessagePromptTemplate, StringPromptTemplate]:
        if self.role=="placeholder":
            return self.template
        else:
            final_template_string = self.template
            if self.partial_variables_builders:
                
                for final_partial_key, partial_builder in self.partial_variables_builders.items():
                    final_partial_value = partial_builder(input_variable_values)
                    final_template_string=final_template_string.replace(f"{{{final_partial_key}}}",final_partial_value)
            
            return final_template_string
            

def build_template_drafts(template:str, format:str, role:str=None )->PromptTemplateDraft:
    partials_with_params={}
    
    if role !="placeholder" and format=="f-string-extra":
        optional_blocks_regex = list(re.finditer(r"\{\?(?P<optional_partial>.+?)(?=\?\})\?\}", template, re.MULTILINE | re.DOTALL))
        for optional_block in optional_blocks_regex:
            optional_partial = optional_block.group("optional_partial")
            partial_input_variables = {v for _, v, _, _ in Formatter().parse(optional_partial) if v is not None}
            
            if not partial_input_variables:
                raise ValueError(f"Optional partial {optional_partial} does not contain any optional variables. Didn't you forget to wrap your parameter in {{}}?")
            
            
            # replace  {} with [] and all other non-word characters with underscore
            partial_name = re.sub(r"[^\w\[\]]+", "_", optional_partial.replace("{","[").replace("}","]"))
            

            partials_with_params[partial_name] = (optional_partial, partial_input_variables)
            # replace optional partial with a placeholder
            template = template.replace(optional_block.group(0), f"{{{partial_name}}}")

        partial_builders = {} # partial_name: a function that takes in a dict of variables and returns a string...
        
        
                
        for partial_name, (partial, partial_input_variables) in partials_with_params.items():
            # create function that will render the partial if all the input variables are present. Otherwise, it will return an empty string... 
            # it needs to be unique for each partial, since we check only for the variables that are present in the partial
            def partial_formatter(inputs, _partial=partial, _partial_input_variables=partial_input_variables):
                """ This will render the partial if all the input variables are present. Otherwise, it will return an empty string."""
                missing_param = next((param for param in _partial_input_variables if param not in inputs or not inputs[param]), None)
                if missing_param:
                    return ""
                else:
                    return _partial
            
            partial_builders[partial_name] = partial_formatter
    try:
        input_variables = [v for _, v, _, _ in Formatter().parse(template) if v is not None and v not in partials_with_params]
    except ValueError as e:
        raise ValueError(f"{e}\nError parsing template: \n```\n{template}\n```")
    for partial_name, (partial, partial_input_variables) in partials_with_params.items():
        input_variables.extend(partial_input_variables)

    input_variables=list(set(input_variables))

    if not partials_with_params:
        partials_with_params=None
        partial_builders=None
    if not role:
        return PromptTemplateDraft(input_variables=input_variables, template=template, partial_variables_builders=partial_builders)
    elif role=="placeholder":
        if len(input_variables)>1:
            raise ValueError(f"Placeholder prompt can only have one input variable, got {input_variables}")
        elif len(input_variables)==0:
            raise ValueError(f"Placeholder prompt must have one input variable, got none.")
        return PromptTemplateDraft(template=template,  input_variables=input_variables,  partial_variables_builders=partial_builders, role="placeholder")
    else:
        return PromptTemplateDraft(role=role, input_variables=input_variables, template=template, partial_variables_builders=partial_builders)
        

def parse_prompt_template_from_string(template_string: str, format: str = "f-string-extra") -> List[PromptTemplateDraft]:
    """Parse a prompt template from a string.
    
    Args:
        template_string (str): The template string to parse.
        format (str, optional): The format of the template. Defaults to "f-string-extra".
        
    Returns:
        List[PromptTemplateDraft]: A list of prompt template drafts.
    """
    prompts = parse_prompts_from_docs(template_string)
    
    if isinstance(prompts, list):
        prompt_template_drafts = []
        
        for prompt in prompts:
            if isinstance(prompt, str):
                prompt_template_drafts.append(build_template_drafts(prompt, format=format))
            else:
                role, content_template = prompt
                message_template = build_template_drafts(content_template, format=format, role=role)
                prompt_template_drafts.append(message_template)
    else:
        prompt_template_drafts = [build_template_drafts(prompts, format=format)]
    
    return prompt_template_drafts

class PromptDecoratorTemplate(StringPromptTemplate):
    template_string: str = ""
    prompt_template_drafts: Union[PromptTemplateDraft, List[PromptTemplateDraft]] = []
    template_name: str = ""
    template_format: str = "f-string-extra"
    optional_variables: List[str] = []
    default_values: Dict[str, Any] = {}
    format_instructions_parameter_key: str = "FORMAT_INSTRUCTIONS"
    template_version: str = "1.0.0"  # Making this optional with a default value
    prompt_type: Optional[PromptTypeSettings] = None
    original_kwargs: Optional[dict] = None

    def __init__(self, template_string: str, template_name: str, template_format: str, input_variables: List[str], 
                 prompt_template_drafts: Union[PromptTemplateDraft, List[PromptTemplateDraft]], 
                 optional_variables: List[str] = None, default_values: Dict[str, Any] = None,
                 format_instructions_parameter_key: str = "FORMAT_INSTRUCTIONS", template_version: str = "1.0.0",
                 prompt_type: PromptTypeSettings = None, original_kwargs: dict = None, **kwargs):
        super().__init__(input_variables=input_variables, **kwargs)
        self.template_string = template_string
        self.template_name = template_name
        self.template_format = template_format
        self.prompt_template_drafts = prompt_template_drafts
        self.optional_variables = optional_variables or []
        self.default_values = default_values or {}
        self.format_instructions_parameter_key = format_instructions_parameter_key
        self.template_version = template_version
        self.prompt_type = prompt_type
        self.original_kwargs = original_kwargs or {}

    @classmethod 
    def build(self, 
             template_string:str, 
             template_name:str=None, 
             template_version:str="1.0.0",
             template_format:str="f-string-extra",
             output_parser:Union[None, BaseOutputParser]=None,
             optional_variables:List[str]=None,
             default_values:Dict[str,Any]=None,
             format_instructions_parameter_key:str="FORMAT_INSTRUCTIONS",
             prompt_type:PromptTypeSettings=None,
             **kwargs
             )->"PromptDecoratorTemplate":
        
        optional_variables = optional_variables or []
        default_values = default_values or {}
        
        if template_format == "f-string-extra":
            prompt_template_drafts = parse_prompt_template_from_string(template_string, format=template_format)
            input_variables = []
            for draft in prompt_template_drafts:
                input_variables.extend(draft.input_variables)
            input_variables = list(set(input_variables))
        else:
            raise ValueError(f"Unsupported template format: {template_format}")
        
        return PromptDecoratorTemplate(
            template_string=template_string,
            template_name=template_name,
            template_version=template_version,
            template_format=template_format,
            input_variables=input_variables,
            optional_variables=optional_variables,
            default_values=default_values,
            output_parser=output_parser,
            format_instructions_parameter_key=format_instructions_parameter_key,
            prompt_type=prompt_type,
            prompt_template_drafts=prompt_template_drafts,
            **kwargs
        )

    @classmethod 
    def from_func(cls, 
                  func:Union[Callable, Coroutine], 
                  template_name:str=None, 
                  template_version:str="1.0.0",
                  output_parser:Union[str,None, BaseOutputParser]="auto", 
                  template_format:str="f-string-extra",
                  format_instructions_parameter_key:str="FORMAT_INSTRUCTIONS",
                  prompt_type:PromptTypeSettings=None,
                  original_kwargs:dict=None
                  )->"PromptDecoratorTemplate":
        
        template_string = get_function_docs(func)  
        template_name = template_name or get_function_full_name(func)
        return_type = get_func_return_type(func)
        original_kwargs = original_kwargs or {}
        
        if original_kwargs.get("output_parser"):
            output_parser = original_kwargs.pop("output_parser")
            
        if output_parser == "auto":
            output_parser = get_output_parser_for_type(return_type)
            
        if isinstance(output_parser, str):
            if output_parser == "str":
                output_parser = None
            elif output_parser == "json":
                output_parser = JsonOutputParser()
            elif output_parser == "boolean":
                output_parser = BooleanOutputParser()
            elif output_parser == "markdown":
                if return_type and return_type != dict:
                    raise Exception(f"Conflicting output parsing instructions. Markdown output parser only supports return type dict, got {return_type}.")
                else:
                    output_parser = MarkdownStructureParser()
            elif output_parser == "list":
                output_parser = ListOutputParser()
            elif output_parser == "pydantic":
                if issubclass(return_type, BaseModel):
                    output_parser = PydanticOutputParser(model=return_type)
                elif return_type is None:
                    raise Exception("You must annotate the return type for pydantic output parser, so that we can infer the model")
                else:
                    raise Exception(f"Unsupported return type {return_type} for pydantic output parser")
            elif output_parser == "functions":
                if not return_type:
                    raise Exception("You must annotate the return type for functions output parser, so that we can infer the model")
                elif not issubclass(return_type, OutputWithFunctionCall):
                    if issubclass(return_type, BaseModel):
                        output_parser = OpenAIFunctionsPydanticOutputParser(model=return_type)
                    else:
                        raise Exception(f"Functions output parser only supports return type pydantic models, got {return_type}")
                else:
                    output_parser = None
            else:
                raise Exception(f"Unsupported output parser {output_parser}")
            
        return cls.build(
            template_string=template_string,
            template_name=template_name,
            template_version=template_version,
            template_format=template_format,
            output_parser=output_parser,
            format_instructions_parameter_key=format_instructions_parameter_key,
            prompt_type=prompt_type,
            **original_kwargs
        )


    def get_final_template(self, **kwargs: Any)->PromptTemplate:
        """Create Chat Messages."""
        
        prompt_type = self.prompt_type or PromptTypeSettings()
        
        if self.default_values:
            # if we have default values, we will use them to fill in missing values
            kwargs = {**self.default_values, **kwargs}

        kwargs = {k:v for k,v in kwargs.items() if v is not None}

        parts=[]
        if isinstance(self.prompt_template_drafts,list):
            
            for message_draft in self.prompt_template_drafts:
                msg_template_final_str = message_draft.finalize_template(kwargs)
                if msg_template_final_str: # skip empty messages / templates
                    parts.append((msg_template_final_str,message_draft.role))

        else:
            msg_template_final_str = self.prompt_template_drafts.finalize_template(kwargs)
            parts.append((msg_template_final_str,""))
        
        template = prompt_type.prompt_template_builder.build_template(parts, kwargs)
        template.output_parser = self.output_parser
            
            
        register_prompt_template(self.template_name, template, self.template_version)
        return template

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        if self.format_instructions_parameter_key in self.input_variables and  not kwargs.get(self.format_instructions_parameter_key)  and self.output_parser :
            # add format instructions to inputs
            kwargs[self.format_instructions_parameter_key] = self.output_parser.get_format_instructions()
            
        final_template = self.get_final_template(**kwargs)
        kwargs = {k:(v if v is not None else "" ) for k,v in  kwargs.items() if k in  final_template.input_variables}
        if isinstance(final_template,ChatPromptTemplate):

            for msg in list(final_template.messages):
                if isinstance(msg,MessagesPlaceholder):
                    if not kwargs.get(msg.variable_name):
                        kwargs[msg.variable_name] = []
        
        for key, value in list(kwargs.items()):
            if isinstance(value, BaseMemory):
                memory:BaseMemory = kwargs.pop(key)
                
                kwargs.update(memory.load_memory_variables(kwargs))
            elif isinstance(value, BaseChatMessageHistory):
                kwargs[key] = value.messages

        formatted =  final_template.format_prompt(**kwargs)
        if isinstance(formatted,ChatPromptValue):
            for msg in list(formatted.messages):
                if (not msg.content or not msg.content.strip() )and not msg.additional_kwargs:
                    formatted.messages.remove(msg)
        self.on_prompt_formatted(formatted.to_string())

        return formatted
    
    
    def format(self, **kwargs: Any) ->str:
        formatted = self.get_final_template(**kwargs).format(**kwargs)
        self.on_prompt_formatted(formatted)
        return formatted

    def on_prompt_formatted(self, formatted:str):
        if not self.prompt_type :
            log_level = logging.DEBUG
        else:
            log_level = self.prompt_type.log_level
            
        log_color =  LogColors.DARK_GRAY # we dont want to color the prompt, is's misleading... we color only the output
        print_log(f"Prompt:\n{formatted}",  log_level , log_color)

def get_output_parser_for_type(return_type):
    if return_type == str or return_type is None:
        return None
    elif return_type == dict:
        return JsonOutputParser()
    elif return_type == list:
        _, args = get_func_return_type(return_type, with_args=True)
        if not args:
            return ListOutputParser()
        if issubclass(args[0], BaseModel):
            return PydanticOutputParser(model=args[0])
        elif issubclass(args[0], dict):
            return JsonOutputParser()
        elif issubclass(args[0], str):
            return ListOutputParser()
        else:
            raise Exception(f"Unsupported item type in list annotation: {args[0]}")
    elif return_type == bool:
        return BooleanOutputParser()
    elif issubclass(return_type, OutputWithFunctionCall):
        return None
    elif issubclass(return_type, BaseModel):
        return PydanticOutputParser(model=return_type)
    else:
        raise Exception(f"Unsupported return type {return_type}")


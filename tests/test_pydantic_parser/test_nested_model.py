
from pydantic import BaseModel, Field
from typing import Union, Optional, List
from langchain_decorators.output_parsers import PydanticOutputParser
from datetime import datetime

class ChildModel1(BaseModel):
    field1: str = Field(description="A string value representing some text data")
    field2: int = Field(description="An integer value representing a count or id")

class ChildModel2(BaseModel):
    field3: float = Field(description="A floating point value representing a measurement")
    field4: bool = Field(description="A boolean value indicating on/off state")

class OuterModel(BaseModel):
    name: str = Field(description="The name of the model instance")
    description: Optional[str] = Field(description="An optional description of the model instance", default=None)
    data: List[Union[ChildModel1, ChildModel2]] = Field(description="The data payload which can be either ChildModel1 or ChildModel2")
    child_model: ChildModel1 = Field(description="The child model instance")
    


class NestedModel(BaseModel):
    value: str = Field(description="A string value")
    timestamp: datetime = Field(description="When this was created")

class ComplexModel(BaseModel):
    complex_field: Optional[List[Union[str, NestedModel, dict]]] = Field(
        description="A complex field that can be a list of either strings or nested models"
    )
    outer_model: OuterModel = Field(description="An outer model instance")

def test_PydanticOutputParser_get_json_example_description():
    parser = PydanticOutputParser(model=ComplexModel, instructions_as_json_example=True)
    
    expected_format_instructions = """
{
\t"complex_field": A complex field that can be a list of either strings or nested models as [optional] valid JSON array
\t\t[
\t\t\t\tstring
\t
\t\t\t\tor
\t
\t\t\t\t
\t\t\t\t{
\t\t\t\t\t"value": A string value as string,
\t\t\t\t\t"timestamp": When this was created as <class 'datetime.datetime'>
\t\t\t\t}
\t
\t\t\t\tor
\t
\t\t\t\tvalid JSON object
\t\t],
\t"outer_model": An outer model instance as 
\t\t{
\t\t\t"name": The name of the model instance as string,
\t\t\t"description": An optional description of the model instance as [optional] string,
\t\t\t"data": The data payload which can be either ChildModel1 or ChildModel2 as valid JSON array
\t\t\t\t\t[
\t\t\t\t\t\t\t
\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t"field1": A string value representing some text data as string,
\t\t\t\t\t\t\t\t"field2": An integer value representing a count or id as int
\t\t\t\t\t\t\t}
\t\t\t
\t\t\t\t\t\t\tor
\t\t\t
\t\t\t\t\t\t\t
\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t"field3": A floating point value representing a measurement as float,
\t\t\t\t\t\t\t\t"field4": A boolean value indicating on/off state as bool
\t\t\t\t\t\t\t}
\t\t\t\t\t],
\t\t\t"child_model": The child model instance as 
\t\t\t\t\t{
\t\t\t\t\t\t"field1": A string value representing some text data as string,
\t\t\t\t\t\t"field2": An integer value representing a count or id as int
\t\t\t\t\t}
\t\t}
}"""
    
    format_instructions = parser.get_json_example_description()
    
    assert format_instructions == expected_format_instructions
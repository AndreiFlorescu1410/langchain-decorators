from typing import Any, Optional, Type, Union, get_args, get_origin, Dict, List
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

def get_field_type(field_info: FieldInfo) -> Type:
    """Get the type of a field, handling Optional types."""
    annotation = field_info.annotation
    if get_origin(annotation) is Union:
        # Handle Optional[Type] which is Union[Type, None]
        args = get_args(annotation)
        if len(args) == 2 and args[1] is type(None):
            return args[0]
    return annotation

def is_field_nullable(field_info: FieldInfo) -> bool:
    """Check if a field is nullable (Optional)."""
    annotation = field_info.annotation
    if get_origin(annotation) is Union:
        args = get_args(annotation)
        return len(args) == 2 and args[1] is type(None)
    return False

def get_field_item_type(field_info: FieldInfo) -> Optional[Type]:
    """Get the item type for a collection field (List, Dict, etc)."""
    annotation = field_info.annotation
    origin = get_origin(annotation)
    if origin in (list, List):
        args = get_args(annotation)
        return args[0] if args else Any
    elif origin in (dict, Dict):
        args = get_args(annotation)
        return args[1] if args else Any
    return None

def align_fields_with_model(data: dict, model: Type[BaseModel]) -> dict:
    """Align field names from input data with model field names."""
    res = {}
    data_with_compressed_keys = None
    
    for field_name, field_info in model.model_fields.items():
        value = None
        if field_name in data:
            value = data[field_name]
        elif field_info.title is not None:
            if field_info.title in data:
                value = data[field_info.title]
            elif field_info.title.lower() in data:
                value = data[field_info.title.lower()]
        elif field_info.alias:
            if field_info.alias in data:
                value = data[field_info.alias]
            elif field_info.alias.lower() in data:
                value = data[field_info.alias.lower()]
        else:
            if not data_with_compressed_keys:
                data_with_compressed_keys = {k.lower().replace(" ", ""): v for k, v in data.items()}
            compressed_key = field_name.lower().replace(" ", "").replace("_", "")
            if compressed_key in data_with_compressed_keys:
                value = data_with_compressed_keys[compressed_key]
        
        if isinstance(value, dict):
            field_type = get_origin(field_info.annotation) if field_info.annotation else None
            if field_info.annotation and isinstance(field_type, type) and issubclass(field_type, BaseModel):
                value = align_fields_with_model(value, field_info.annotation)
        elif isinstance(value, list):
            if field_info.annotation:
                item_type = get_field_item_type(field_info)
                if item_type and issubclass(item_type, BaseModel):
                    value = [align_fields_with_model(item, item_type) for item in value]
        res[field_name] = value
    return res

def humanize_pydantic_validation_error(validation_error: ValidationError) -> str:
    """Convert a Pydantic ValidationError into a human-readable string."""
    return "\n".join([f'{".".join([str(i) for i in err.get("loc")])} - {err.get("msg")}' for err in validation_error.errors()])

def sanitize_pydantic_schema(schema: dict) -> dict:
    """Sanitize a Pydantic schema by resolving $ref references."""
    if schema.get("definitions"):
        definitions = schema.pop("definitions")
        for def_key, definition in definitions.items():
            if "title" in definition:
                definition.pop("title")  # no need for this
            nested_ref = next((1 for val in definition.get("properties", {}).values() 
                             if isinstance(val, dict) and val.get("$ref")), None)
            if nested_ref:
                raise Exception(f"Nested $ref not supported! ... probably recursive schema: {def_key}")
        
        def replace_refs_recursive(schema: dict) -> None:
            if isinstance(schema, dict):
                if schema.get("properties"):
                    for k, v in schema["properties"].items():
                        if v.get("$ref"):
                            schema["properties"][k] = definitions[v["$ref"].split("/")[-1]]
                        elif v.get("properties"):
                            replace_refs_recursive(v)
                        elif v.get("items"):
                            if isinstance(v["items"], dict):
                                ref = v["items"].get("$ref")
                                if ref:
                                    v["items"] = definitions[ref.split("/")[-1]]
                if schema.get("items") and schema["items"].get("$ref"):
                    ref = schema["items"]["$ref"]
                    ref_key = ref.split("/")[-1]
                    schema["items"] = definitions.get(ref_key)

        replace_refs_recursive(schema)
    return schema

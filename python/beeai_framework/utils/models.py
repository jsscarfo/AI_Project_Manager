#!/usr/bin/env python
# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for working with data models.
"""

import inspect
from abc import ABC
from collections.abc import Sequence
from contextlib import suppress
from logging import Logger
from typing import Any, Dict, Literal, Optional, Type, TypeVar, Union, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, GetJsonSchemaHandler, create_model
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, SchemaValidator

logger = Logger(__name__)

T = TypeVar("T", bound=BaseModel)
ModelLike = Union[T, dict[str, Any]]  # noqa: UP007


def to_model(cls: type[T], obj: ModelLike[T]) -> T:
    return obj if isinstance(obj, cls) else cls.model_validate(obj, strict=False, from_attributes=True)


def to_any_model(classes: Sequence[type[BaseModel]], obj: ModelLike[T]) -> Any:
    if len(classes) == 1:
        return to_model(classes[0], obj)

    for cls in classes:
        with suppress(Exception):
            return to_model(cls, obj)

    return ValueError(
        "Failed to create a model instance from the passed object!" + "\n".join(cls.__name__ for cls in classes),
    )


def to_model_optional(cls: type[T], obj: ModelLike[T] | None) -> T | None:
    return None if obj is None else to_model(cls, obj)


def check_model(model: T) -> None:
    schema_validator = SchemaValidator(schema=model.__pydantic_core_schema__)
    schema_validator.validate_python(model.__dict__)


class JSONSchemaModel(ABC, BaseModel):
    _custom_json_schema: JsonSchemaValue

    model_config = ConfigDict(
        arbitrary_types_allowed=False, validate_default=True, json_schema_mode_override="validation"
    )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: CoreSchema,
        handler: GetJsonSchemaHandler,
        /,
    ) -> JsonSchemaValue:
        return cls._custom_json_schema.copy()

    @classmethod
    def create(cls, schema_name: str, schema: dict[str, Any]) -> type["JSONSchemaModel"]:
        type_mapping: dict[str, Any] = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict,
            "array": list,
            "null": None,
        }

        fields: dict[str, tuple[type, Any]] = {}
        required = set(schema.get("required", []))
        properties = schema.get("properties", {})

        for param_name, param in properties.items():
            target_type: type | Any = type_mapping.get(param.get("type"))
            is_optional = param_name not in required
            if is_optional:
                target_type = Optional[target_type] if target_type else type(None)  # noqa: UP007

            if isinstance(param.get("const"), str):
                target_type = Literal[param["const"]]
            if not target_type:
                logger.debug(
                    f"{JSONSchemaModel.__name__}: Can't resolve a correct type for '{param_name}' attribute."
                    f" Using 'Any' as a fallback."
                )
                target_type = type

            if target_type is dict:
                target_type = cls.create(param_name, param)

            fields[param_name] = (
                target_type,
                Field(
                    description=param.get("description"),
                    default=None if is_optional else param["const"] if param.get("const") else ...,
                ),
            )

        model: type[JSONSchemaModel] = create_model(  # type: ignore
            schema_name,
            **fields,
            __base__=cls,
        )
        model._custom_json_schema = schema
        return model


def get_schema(model_type: Type[T]) -> Dict[str, Any]:
    """
    Get the JSON schema for a Pydantic model.
    
    Args:
        model_type: The Pydantic model class
        
    Returns:
        The JSON schema as a dictionary
    """
    if hasattr(model_type, "model_json_schema"):
        # Pydantic v2
        return model_type.model_json_schema()
    elif hasattr(model_type, "schema"):
        # Pydantic v1
        return model_type.schema()
    else:
        # Fallback for non-Pydantic types
        schema = {
            "type": "object",
            "properties": {}
        }
        
        if inspect.isclass(model_type):
            # Try to get type hints
            hints = get_type_hints(model_type)
            for name, hint in hints.items():
                if name.startswith('_'):
                    continue
                    
                # Basic type mapping
                if hint == str:
                    schema["properties"][name] = {"type": "string"}
                elif hint == int:
                    schema["properties"][name] = {"type": "integer"}
                elif hint == float:
                    schema["properties"][name] = {"type": "number"}
                elif hint == bool:
                    schema["properties"][name] = {"type": "boolean"}
                elif hint == list or hint == set:
                    schema["properties"][name] = {"type": "array", "items": {}}
                elif hint == dict:
                    schema["properties"][name] = {"type": "object"}
                else:
                    schema["properties"][name] = {"type": "object"}
        
        return schema

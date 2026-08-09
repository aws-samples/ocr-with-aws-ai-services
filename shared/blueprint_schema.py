"""
Translation between a nested JSON output schema and an Amazon BDA blueprint.

A BDA blueprint is not free-form JSON Schema. It supports a single level of
nesting, expressed with `$ref` pointers into a top-level `definitions` block:

  * a nested object ("group")        -> {"$ref": "#/definitions/X"}
  * an array of objects ("table")    -> {"type": "array", "items": {"$ref": ...}}
  * an array of scalars              -> {"type": "array", "items": {"type": ...}}

An array of objects *inside* a group is rejected by `CreateBlueprint`
("Request has invalid blueprint schema"), so such a field is hoisted to the top
level and put back afterwards. Everything the blueprint returns therefore has to
be mapped back onto the original nested paths before it can be compared against
ground truth, which is what `restore_nested_result` does.

The mapping is recorded at build time and is the only authority for where a
returned key belongs. Blueprint field names are never taken apart on a separator:
a schema may legitimately contain a field called `part_a`, so splitting a name
like `part_a__rows` back into segments would be guesswork.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from shared.config import logger

# Separator used when a field has to be hoisted out of its parent group. Chosen
# only for readability in the blueprint listing - `field_map` is what actually
# resolves a name back to its path, so this string carries no meaning.
HOIST_SEPARATOR = "__"

# JSON Schema scalar types a blueprint field may declare.
_BLUEPRINT_SCALAR_TYPES = ("string", "number", "boolean", "integer")

# BDA distinguishes values that are written on the page from values that have to
# be derived. Everything produced here is read directly off the document.
_INFERENCE_TYPE = "explicit"


@dataclass
class BlueprintBuild:
    """
    The result of translating an output schema into a BDA blueprint.

    Attributes:
        schema: Blueprint JSON ready to pass to `CreateBlueprint`.
        field_map: Blueprint top-level key -> path that key occupies in the
            original schema, as a tuple of segments.
        field_count: Number of billable extraction fields in the blueprint.
        hoisted_paths: Paths that had to be moved to the top level because BDA
            cannot nest them, kept so the UI and the log can report it.
    """

    schema: Dict[str, Any]
    field_map: Dict[str, Tuple[str, ...]]
    field_count: int
    hoisted_paths: List[Tuple[str, ...]] = field(default_factory=list)


def resolve_scalar_type(*, declared_type: Any) -> str:
    """
    Reduce a JSON Schema type declaration to the single scalar type BDA accepts.

    JSON Schema allows a union such as ["number", "null"] to mark an optional
    field. A blueprint field takes one type, so the first non-null member wins.

    Args:
        declared_type: The schema's `type` value - a string, or a list of
            strings for a union type.

    Returns:
        One of "string", "number", "boolean" or "integer".
    """
    if isinstance(declared_type, list):
        # "null" only signals optionality; it is not a type BDA can extract.
        for candidate in declared_type:
            if candidate != "null":
                return resolve_scalar_type(declared_type=candidate)
        raise ValueError(
            f"Type declaration {declared_type!r} names no extractable type"
        )

    if declared_type in _BLUEPRINT_SCALAR_TYPES:
        return declared_type

    # An unrecognised or absent type is treated as text rather than dropped: the
    # field still exists on the page and a string is always extractable.
    return "string"


def _scalar_field(*, name: str, definition: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a blueprint scalar field from a schema property.

    Args:
        name: Property name, used only for the fallback instruction.
        definition: The schema property object.

    Returns:
        A blueprint field definition.
    """
    return {
        "type": resolve_scalar_type(declared_type=definition.get("type")),
        "inferenceType": _INFERENCE_TYPE,
        "instruction": definition.get("description", f"Extract the {name}"),
    }


def _scalar_array_field(*, name: str, definition: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a blueprint array-of-scalars field from a schema property.

    Args:
        name: Property name, used only for the fallback instruction.
        definition: The schema property object, whose `items` is a scalar.

    Returns:
        A blueprint field definition.
    """
    items = definition.get("items") or {}
    return {
        "type": "array",
        "inferenceType": _INFERENCE_TYPE,
        "instruction": definition.get("description", f"Extract all {name} values"),
        "items": {"type": resolve_scalar_type(declared_type=items.get("type"))},
    }


def _is_object(*, definition: Dict[str, Any]) -> bool:
    """
    Report whether a schema property describes an object with named properties.

    Args:
        definition: The schema property object.

    Returns:
        True when the property is a group of named sub-properties.
    """
    return definition.get("type") == "object" and isinstance(
        definition.get("properties"), dict
    )


def _is_object_array(*, definition: Dict[str, Any]) -> bool:
    """
    Report whether a schema property describes an array of objects (a table).

    Args:
        definition: The schema property object.

    Returns:
        True when the property is an array whose items are objects with named
        properties.
    """
    if definition.get("type") != "array":
        return False
    items = definition.get("items")
    return isinstance(items, dict) and _is_object(definition=items)


def _definition_name(*, path: Tuple[str, ...], taken: Set[str]) -> str:
    """
    Pick an unused name for a `definitions` entry.

    Args:
        path: Path of the field the definition describes.
        taken: Names already used in the definitions block.

    Returns:
        A name unique within the definitions block.
    """
    base = "".join(segment[:1].upper() + segment[1:] for segment in path)
    name = base
    suffix = 2
    while name in taken:
        name = f"{base}{suffix}"
        suffix += 1
    taken.add(name)
    return name


def _table_field(
    *,
    name: str,
    definition: Dict[str, Any],
    path: Tuple[str, ...],
    definitions: Dict[str, Any],
    definition_names: Set[str],
) -> Tuple[Dict[str, Any], int]:
    """
    Build a blueprint table field, registering its row shape in `definitions`.

    Args:
        name: Property name, used only for the fallback instruction.
        definition: The schema property object - an array of objects.
        path: Path of the property in the original schema, used to name the
            definitions entry.
        definitions: The blueprint's definitions block, mutated in place.
        definition_names: Names already used in that block, mutated in place.

    Returns:
        Tuple of (blueprint field definition, number of columns it declares).
    """
    columns = definition["items"]["properties"]
    row_name = _definition_name(path=path, taken=definition_names)

    row_properties: Dict[str, Any] = {}
    for column_name, column in columns.items():
        if _is_object(definition=column) or _is_object_array(definition=column):
            # A table column has to be a scalar. Extracting it as text keeps the
            # value rather than dropping the column, but it will not match a
            # nested truth value, so say so.
            logger.warning(
                f"Table column {'.'.join(path + (column_name,))} is not a scalar; "
                f"extracting it as text, which will not match a nested value"
            )
        row_properties[column_name] = _scalar_field(
            name=column_name, definition=column
        )

    definitions[row_name] = {"properties": row_properties}

    table = {
        "type": "array",
        "instruction": definition.get("description", f"Each row of {name}"),
        "items": {"$ref": f"#/definitions/{row_name}"},
    }
    return table, len(row_properties)


def _blueprint_key(*, path: Tuple[str, ...]) -> str:
    """
    Build the top-level blueprint field name for a path.

    A depth-1 path keeps its own name. Anything deeper is being hoisted out of
    its parent, and the joined name exists only so the blueprint listing reads
    sensibly - `field_map` is what resolves it back.

    Args:
        path: Path of the field in the original schema.

    Returns:
        The blueprint's top-level key for that path.
    """
    return HOIST_SEPARATOR.join(path)


@dataclass
class _Emitter:
    """
    Accumulates blueprint state while walking a nested schema.

    Attributes:
        properties: Blueprint top-level properties, built up in place.
        definitions: Blueprint definitions block, built up in place.
        definition_names: Definition names already used.
        field_map: Blueprint key -> original path.
        hoisted_paths: Paths moved to the top level because BDA cannot nest them.
        source_keys: Top-level keys of the original schema, so a generated hoist
            name that would shadow one can be rejected.
        field_count: Billable extraction fields emitted so far.
    """

    properties: Dict[str, Any] = field(default_factory=dict)
    definitions: Dict[str, Any] = field(default_factory=dict)
    definition_names: Set[str] = field(default_factory=set)
    field_map: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    hoisted_paths: List[Tuple[str, ...]] = field(default_factory=list)
    source_keys: Set[str] = field(default_factory=set)
    field_count: int = 0

    def claim(self, *, path: Tuple[str, ...]) -> str:
        """
        Reserve the top-level blueprint key for a path.

        Args:
            path: Path of the field being emitted.

        Returns:
            The reserved key.

        Raises:
            ValueError: If the key is already taken by another field, which would
                otherwise silently overwrite one of them.
        """
        key = _blueprint_key(path=path)
        hoisted = len(path) > 1
        if key in self.properties or (hoisted and key in self.source_keys):
            raise ValueError(
                f"Cannot hoist {'.'.join(path)} to top-level field '{key}': "
                f"that name is already taken"
            )
        if hoisted:
            self.hoisted_paths.append(path)
        return key


def _emit_container(
    *, path: Tuple[str, ...], definition: Dict[str, Any], emitter: _Emitter
) -> None:
    """
    Emit one object or array-of-objects, hoisting anything BDA cannot nest.

    BDA blueprints allow a single level of nesting: a group of scalars, or a
    table of scalar columns. A container found inside a group is therefore
    emitted as its own top-level field and recorded in `field_map` so
    `restore_nested_result` can put it back where the schema wants it.

    Args:
        path: Path of this container in the original schema.
        definition: The container's schema property object.
        emitter: Accumulated blueprint state, mutated in place.
    """
    if _is_object_array(definition=definition):
        table, column_count = _table_field(
            name=path[-1],
            definition=definition,
            path=path,
            definitions=emitter.definitions,
            definition_names=emitter.definition_names,
        )
        key = emitter.claim(path=path)
        emitter.properties[key] = table
        emitter.field_map[key] = path
        emitter.field_count += column_count
        return

    group_properties: Dict[str, Any] = {}
    nested_containers: List[Tuple[Tuple[str, ...], Dict[str, Any]]] = []

    for child_name, child in definition["properties"].items():
        if not isinstance(child, dict):
            continue
        child_path = path + (child_name,)

        if _is_object(definition=child) or _is_object_array(definition=child):
            # Deferred rather than recursed into immediately, so this group's own
            # key is claimed before the children that hoist out of it.
            nested_containers.append((child_path, child))
        elif child.get("type") == "array":
            group_properties[child_name] = _scalar_array_field(
                name=child_name, definition=child
            )
            emitter.field_count += 1
        else:
            group_properties[child_name] = _scalar_field(
                name=child_name, definition=child
            )
            emitter.field_count += 1

    if group_properties:
        group_name = _definition_name(path=path, taken=emitter.definition_names)
        emitter.definitions[group_name] = {"properties": group_properties}
        key = emitter.claim(path=path)
        group_field: Dict[str, Any] = {"$ref": f"#/definitions/{group_name}"}
        if definition.get("description"):
            group_field["instruction"] = definition["description"]
        emitter.properties[key] = group_field
        emitter.field_map[key] = path

    for child_path, child in nested_containers:
        _emit_container(path=child_path, definition=child, emitter=emitter)


def build_blueprint_schema(
    *, schema: Any, document_type: str = "generic"
) -> Optional[BlueprintBuild]:
    """
    Translate a nested output schema into a BDA blueprint.

    Nested objects become groups and arrays of objects become tables, both via
    `$ref` into a `definitions` block. BDA supports only one level of nesting, so
    a container found inside a group - whether an object or a table - is hoisted
    to the top level and recorded in `field_map` so the result can be put back.

    Args:
        schema: The output schema, as a JSON string or an already-parsed dict.
        document_type: Document class recorded on the blueprint.

    Returns:
        A `BlueprintBuild`, or None when the schema cannot be parsed or declares
        no properties.
    """
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except ValueError as parse_error:
            logger.error(f"Output schema is not valid JSON: {parse_error}")
            return None

    if not isinstance(schema, dict) or not isinstance(schema.get("properties"), dict):
        logger.error("Output schema declares no properties; cannot build a blueprint")
        return None

    emitter = _Emitter(source_keys=set(schema["properties"]))

    for prop_name, prop in schema["properties"].items():
        if not isinstance(prop, dict):
            continue
        path = (prop_name,)

        if _is_object(definition=prop) or _is_object_array(definition=prop):
            _emit_container(path=path, definition=prop, emitter=emitter)
        elif prop.get("type") == "array":
            emitter.properties[prop_name] = _scalar_array_field(
                name=prop_name, definition=prop
            )
            emitter.field_map[prop_name] = path
            emitter.field_count += 1
        else:
            emitter.properties[prop_name] = _scalar_field(
                name=prop_name, definition=prop
            )
            emitter.field_map[prop_name] = path
            emitter.field_count += 1

    if not emitter.properties:
        logger.error("Output schema produced no blueprint fields")
        return None

    blueprint: Dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "description": f"{document_type.capitalize()} document schema",
        "class": document_type,
        "type": "object",
    }
    # `definitions` is placed before `properties` only so the blueprint listing
    # reads top-down; BDA does not care about key order.
    if emitter.definitions:
        blueprint["definitions"] = emitter.definitions
    blueprint["properties"] = emitter.properties

    return BlueprintBuild(
        schema=blueprint,
        field_map=emitter.field_map,
        field_count=emitter.field_count,
        hoisted_paths=emitter.hoisted_paths,
    )


def restore_nested_result(
    *, inference_result: Dict[str, Any], field_map: Dict[str, Tuple[str, ...]]
) -> Dict[str, Any]:
    """
    Put a blueprint's flat top-level result back into the schema's nested shape.

    Groups and tables are returned by BDA already shaped, so each returned key is
    placed at the path `field_map` recorded for it. Keys the map does not know
    about are kept at the top level rather than dropped, so an unexpected
    addition to the blueprint stays visible instead of vanishing.

    Args:
        inference_result: The `inference_result` object from BDA custom output.
        field_map: Blueprint key -> original path, from `BlueprintBuild`.

    Returns:
        The result nested the way the original output schema declares it.
    """
    restored: Dict[str, Any] = {}

    for key, value in inference_result.items():
        path = field_map.get(key)

        if path is None:
            logger.warning(
                f"Blueprint returned '{key}', which is not in the field map; "
                f"keeping it at the top level"
            )
            restored[key] = value
            continue

        target = restored
        for segment in path[:-1]:
            existing = target.get(segment)
            if not isinstance(existing, dict):
                # Only overwrite when nothing usable is there. A group and a
                # field hoisted out of it both write into the same parent, and
                # whichever arrives second must not erase the first.
                existing = {}
                target[segment] = existing
            target = existing

        leaf = path[-1]
        if isinstance(target.get(leaf), dict) and isinstance(value, dict):
            # The group arrived after a field hoisted out of it. Merge rather
            # than replace so the hoisted field survives.
            target[leaf].update(value)
        else:
            target[leaf] = value

    return restored

"""Compile the verified action schema subset to JSON without formatting whitespace."""

import json


def has_formatting_whitespace(text: str) -> bool:
    """Find whitespace outside JSON strings, preserving spaces and escapes within them."""
    quoted = escaped = False
    for character in text:
        if escaped:
            escaped = False
        elif quoted and character == "\\":
            escaped = True
        elif character == '"':
            quoted = not quoted
        elif not quoted and character.isspace():
            return True
    return False


def compact_json_grammar(schema: dict) -> str:
    """Return bounded EBNF; reject unsupported constraints rather than dropping them.

    Property order follows the requested schema. Spaces inside strings remain
    valid; whitespace between fields is absent from the language.
    """
    definitions = schema.get("$defs", {})
    rules: dict[str, str] = {}
    cached: dict[str, str] = {}
    supported = {
        "$defs", "$ref", "title", "description", "default", "type", "const", "enum",
        "anyOf", "properties", "required", "additionalProperties", "items",
        "prefixItems", "minItems", "maxItems", "minLength", "maxLength",
    }

    def terminal(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    def literal(value) -> str:
        return terminal(json.dumps(value, ensure_ascii=False, separators=(",", ":")))

    def compile_node(node: dict) -> str:
        unknown = set(node) - supported
        if unknown:
            raise ValueError(f"Unsupported compact JSON constraints: {sorted(unknown)}")
        if "$ref" in node:
            ref = node["$ref"]
            if not ref.startswith("#/$defs/"):
                raise ValueError("Compact JSON requires local schema references")
            return compile_node(definitions[ref.removeprefix("#/$defs/")])
        key = json.dumps(node, ensure_ascii=False)
        if key in cached:
            return cached[key]
        name = f"rule{len(cached)}"
        cached[key] = name
        if "const" in node:
            expression = literal(node["const"])
        elif "enum" in node:
            expression = " | ".join(literal(value) for value in node["enum"])
        elif "anyOf" in node:
            expression = " | ".join(compile_node(branch) for branch in node["anyOf"])
        elif node.get("type") == "object":
            props = node.get("properties", {})
            if node.get("additionalProperties") is not False or set(node.get("required", [])) != set(props):
                raise ValueError("Compact JSON requires closed objects with all fields required")
            members = [literal(key) + ' ":" ' + compile_node(value)
                       for key, value in props.items()]
            expression = '"{" ' + ' "," '.join(members) + ' "}"'
        elif node.get("type") == "array":
            prefix = [compile_node(item) for item in node.get("prefixItems", [])]
            item = compile_node(node["items"])
            minimum, maximum = node.get("minItems", 0), node.get("maxItems")
            if maximum is not None:
                if maximum < minimum:
                    raise ValueError("Invalid compact JSON array bounds")
                alternatives = []
                for size in range(minimum, maximum + 1):
                    members = [prefix[index] if index < len(prefix) else item for index in range(size)]
                    alternatives.append('"[" ' + ' "," '.join(members) + ' "]"')
                expression = " | ".join(alternatives)
            elif prefix:
                raise ValueError("Compact JSON requires a maximum for prefix arrays")
            elif minimum == 0:
                expression = f'"[" ({item} ("," {item})*)? "]"'
            else:
                expression = '"[" ' + ' "," '.join([item] * minimum) + f' ("," {item})* "]"'
        elif node.get("type") == "string":
            minimum, maximum = node.get("minLength", 0), node.get("maxLength")
            if maximum is None:
                characters = " ".join(["json-char"] * minimum) + " json-char*"
            elif maximum >= minimum:
                characters = f"json-char{{{minimum},{maximum}}}"
            else:
                raise ValueError("Invalid compact JSON string bounds")
            expression = f'{terminal(chr(34))} {characters} {terminal(chr(34))}'
        elif node.get("type") == "null":
            expression = '"null"'
        elif node.get("type") == "boolean":
            expression = '"true" | "false"'
        else:
            raise ValueError(f"Unsupported compact JSON type: {node.get('type')}")
        rules[name] = expression
        return name

    root = compile_node(schema)
    character = r'json-char ::= [^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})'
    return "\n".join([f"root ::= {root}", character,
                      *(f"{name} ::= {expression}" for name, expression in rules.items())]) + "\n"

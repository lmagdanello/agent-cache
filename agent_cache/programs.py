from __future__ import annotations

import re
from dataclasses import dataclass, field
from string import Formatter
from typing import Iterable

from .models import GeneratedProgram


@dataclass(slots=True)
class ProgramExecutionResult:
    passed: bool
    output: dict[str, object] | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class ExtractedProgramFields:
    fields: dict[str, str] = field(default_factory=dict)
    reasons: tuple[str, ...] = field(default_factory=tuple)


class SafeProgramDSL:
    """
    Safe generated-program helper.

    The DSL supports:
    - extracting named fields from a prompt template
    - applying small deterministic transforms
    - rendering a response template
    - rejecting when required data is missing
    """

    def compile_prompt_pattern(self, template: str) -> re.Pattern[str]:
        formatter = Formatter()
        pattern_parts: list[str] = ["^"]
        for literal_text, field_name, format_spec, conversion in formatter.parse(template):
            if literal_text:
                pattern_parts.append(re.escape(literal_text))
            if field_name is not None:
                pattern_parts.append(fr"(?P<{field_name}>.+?)")
        pattern_parts.append("$")
        return re.compile("".join(pattern_parts), re.IGNORECASE)

    def extract_fields(self, prompt: str, template: str) -> ExtractedProgramFields:
        pattern = self.compile_prompt_pattern(template)
        match = pattern.match(prompt.strip())
        if not match:
            return ExtractedProgramFields({}, ("prompt does not match program template",))
        return ExtractedProgramFields({key: value.strip() for key, value in match.groupdict().items()}, ())

    def apply_transformations(self, fields: dict[str, str], transformations: dict[str, str]) -> dict[str, str]:
        output = dict(fields)
        for field_name, transform in transformations.items():
            if field_name not in output:
                continue
            value = output[field_name]
            output[field_name] = self._apply_transform(value, transform)
        return output

    def render(self, template: str, fields: dict[str, str]) -> str:
        return template.format(**fields)

    def _apply_transform(self, value: str, transform: str) -> str:
        if transform == "lower":
            return value.lower()
        if transform == "upper":
            return value.upper()
        if transform == "title":
            return value.title()
        if transform == "strip":
            return value.strip()
        if transform.startswith("prefix:"):
            return transform.split(":", 1)[1] + value
        if transform.startswith("suffix:"):
            return value + transform.split(":", 1)[1]
        return value


class GeneratedProgramRunner:
    def __init__(self, dsl: SafeProgramDSL | None = None):
        self.dsl = dsl or SafeProgramDSL()

    def execute(self, program: GeneratedProgram, fields: dict[str, str]) -> ProgramExecutionResult:
        missing = [field for field in program.required_fields if field not in fields or fields[field] == ""]
        if missing:
            return ProgramExecutionResult(False, None, (f"missing required fields: {', '.join(missing)}",))

        allowed = set(program.allowed_fields) if program.allowed_fields else set(fields.keys())
        unknown = sorted(set(fields) - allowed)
        if unknown:
            return ProgramExecutionResult(False, None, (f"unknown fields: {', '.join(unknown)}",))

        rendered_fields = self.dsl.apply_transformations(fields, program.transformations)
        try:
            rendered = self.dsl.render(program.template, rendered_fields)
        except KeyError as exc:
            return ProgramExecutionResult(False, None, (f"missing field {exc.args[0]}",))
        except Exception as exc:  # pragma: no cover - defensive
            return ProgramExecutionResult(False, None, (f"template error: {exc}",))
        return ProgramExecutionResult(True, {"text": rendered}, ())


class FieldExtractor:
    def __init__(self, dsl: SafeProgramDSL | None = None):
        self.dsl = dsl or SafeProgramDSL()

    def extract(self, prompt: str, pattern: str) -> dict[str, str]:
        extracted = self.dsl.extract_fields(prompt, pattern)
        return extracted.fields


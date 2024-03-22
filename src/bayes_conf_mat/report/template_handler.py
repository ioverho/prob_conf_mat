import re
import typing
from pathlib import Path
from dataclasses import dataclass, field
from collections import OrderedDict

from bayes_conf_mat.report.utils import fmt

REPL_STRING = re.compile(r"@@(.+?)@@")
TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class KwargMatch:
    key: str = None
    spans: typing.List[typing.Tuple[int, int]] = field(default_factory=lambda: [])
    value: typing.Optional[typing.Any] = None

    def __add__(self, other):
        if self.key is None:
            self.key = other.key

        self.spans = self.spans + other.spans

        return self


class Template:
    def __init__(self, file_name: str) -> None:
        self.file_path = TEMPLATE_DIR / file_name
        self.name = self.file_path.stem

        self.template = self.file_path.read_text()

        self.kwargs = OrderedDict()
        for match in REPL_STRING.finditer(self.template):
            self.kwargs[match.group(1)] = self.kwargs.get(
                match.group(1), KwargMatch()
            ) + KwargMatch(key=match.group(1), spans=[match.span()])

    def set(self, key: str, value):
        self.kwargs[key].value = value

    def _format_value(self, value):
        if isinstance(value, str):
            return value
        elif isinstance(value, float):
            return fmt(value)
        elif isinstance(value, int):
            return f"{value:d}"
        else:
            try:
                return str(value)
            except Exception as e:
                raise TypeError(
                    f"Unable to format {value} of type {type(value)}. Raises {e}"
                )

    def __repr__(self) -> str:
        return f"Template({self.name})"

    def __str__(self) -> str:
        filled_template = ""

        unsorted_spans = [
            (span, kwarg_match.key, kwarg_match.value)
            for kwarg_match in self.kwargs.values()
            for span in kwarg_match.spans
        ]
        sorted_spans = sorted(unsorted_spans, key=lambda x: x[0][0])

        left_pointer = 0
        for (begin, end), key, value in sorted_spans:
            filled_template += self.template[left_pointer:begin]

            if value is not None:
                filled_template += self._format_value(value)
            else:
                filled_template += f"@@{key}@@"

            left_pointer = end

        filled_template += self.template[left_pointer:]

        return filled_template

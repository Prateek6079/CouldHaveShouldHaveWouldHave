"""Reusable token offsets, non-overlapping spans and compiled component grammars."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

from lark import Lark, Token as GrammarToken
from lark.exceptions import UnexpectedInput

TOKEN = re.compile(r"[a-zA-Z]+(?:\.[a-zA-Z]+)+\.?|[^\W_]+(?:['’/\-][^\W_]+)*\.?|[^\s]", re.UNICODE)

STARTS = {
    'us': ['unit', 'floor', 'building', 'locality', 'street', 'landmark', 'po_box', 'postal_code', 'address'],
    'india': ['unit', 'house_no', 'floor', 'building', 'locality', 'district', 'village_component',
              'street', 'landmark', 'po_box', 'postal_code', 'address'],
}

@dataclass(frozen=True, slots=True)
class Token:
    text: str
    start: int
    end: int

def tokenize(text: str) -> list[Token]:
    return [Token(m[0].lower().rstrip('.'), m.start(), m.end()) for m in TOKEN.finditer(text)]

@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    kind: str
    values: dict[str, str]

@lru_cache(maxsize=4)
def grammar(country: str) -> Lark:
    if country not in STARTS:
        raise ValueError(f'No component grammar for {country!r}')
    path = Path(__file__).resolve().parents[1] / f'resources/{country}_address.lark'
    return Lark(path.read_text(encoding='utf-8'), parser='lalr', start=STARTS[country])

def accepts(start: str, typed: list[tuple[str, str]], *, country: str = 'us') -> bool:
    parser = grammar(country).parse_interactive(start=start)
    try:
        for kind, value in typed:
            parser.feed_token(GrammarToken(kind, value))
        parser.feed_eof()
        return True
    except UnexpectedInput:
        return False

class SpanCollector:
    """Only consume successful, nonconflicting components; retain everything else."""
    def __init__(self, text: str):
        self.text = text
        self.tokens = tokenize(text)
        self.claimed = [False] * len(self.tokens)
        self.spans: list[Span] = []
        self.values: dict[str, str] = {}
        self.conflicts: set[str] = set()

    def free(self, start: int, end: int) -> bool:
        return start < end and not any(self.claimed[start:end])

    def add(self, span: Span) -> bool:
        if not self.free(span.start, span.end):
            return False
        conflicts = {k for k, v in span.values.items() if k in self.values and self.values[k] != v}
        if conflicts:
            self.conflicts.update(conflicts)
            return False
        if any(k in self.values for k in span.values):
            return False
        self.values.update(span.values)
        self.spans.append(span)
        self.claimed[span.start:span.end] = [True] * (span.end-span.start)
        return True

    def remainder(self) -> str | None:
        chars = list(self.text)
        for span in self.spans:
            start, end = self.tokens[span.start].start, self.tokens[span.end-1].end
            chars[start:end] = ' ' * (end-start)
        chunks = [' '.join(c.split()).strip(' ,;()') for c in re.split(r'[,;]', ''.join(chars))]
        return ', '.join(c for c in chunks if c) or None

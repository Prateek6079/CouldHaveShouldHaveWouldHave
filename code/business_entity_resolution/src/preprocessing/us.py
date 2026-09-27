"""Shallow US regex + Lark parser with bounded spans and retained unknown text.

Lexical recognizers propose components; the grammar validates their shape and
assembly. An incomplete address keeps every independently recognized component.
The trained usaddress comparison adapter lives in us_legacy.py.
"""
from functools import lru_cache
import json
from pathlib import Path
import re

from .address_schema import AddressResult
from .cities import city_index, key
from .component_grammar import Span, SpanCollector, accepts

NUMBER = re.compile(r'\d+[a-z]?(?:[-/]\d+[a-z]?)?\Z')
ORDINAL = re.compile(r'(\d+)(?:st|nd|rd|th)\Z')
IDENTIFIER = re.compile(r'(?:[a-z]*\d+[a-z]*(?:[-/][a-z0-9]+)?|[a-z]+-\d+[a-z]*|[a-z])\Z')
ZIP = re.compile(r'\d{5}(?:-\d{4})?\Z')
SEPARATORS = {',', ';', '(', ')'}


@lru_cache(maxsize=1)
def lexicon() -> dict:
    path = Path(__file__).resolve().parents[1] / 'resources/us_address_lexicon.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['version'] != 1:
        raise ValueError('Unsupported US address lexicon version')
    return data


def _aliases() -> dict:
    """Compatibility with the address-retention audit."""
    return {'street_types': lexicon()['street'], 'directions': lexicon()['direction']}


class _Parser:
    def __init__(self, text: str):
        self.c = SpanCollector(text)
        self.words = [t.text for t in self.c.tokens]
        self.lex = lexicon()
        self.limit = self.lex['max_component_tokens']

    def phrase(self, start: int, end: int) -> str:
        return ' '.join(self.words[start:end])

    def available(self, start: int, end: int) -> bool:
        return (0 <= start < end <= len(self.words) and end-start <= self.limit
                and self.c.free(start, end)
                and not any(w in SEPARATORS for w in self.words[start:end]))

    def emit(self, start: int, end: int, kind: str, values: dict,
             rule: str | None = None, typed: list | None = None) -> bool:
        if not self.available(start, end):
            return False
        if rule and not accepts(rule, typed):
            return False
        return self.c.add(Span(start, end, kind, values))

    def chunks(self):
        start = None
        for i in range(len(self.words)+1):
            free = i < len(self.words) and not self.c.claimed[i] and self.words[i] not in SEPARATORS
            if free and start is None:
                start = i
            elif not free and start is not None:
                yield start, i
                start = None

    def states(self):
        # A state ends a segment, precedes a ZIP, or occupies its own segment.
        # Prefer explicit codes to full names that can also be city names.
        codes = set(self.lex['states'].values())
        for codes_only in (True, False):
            for i in range(len(self.words)):
                for width in (3, 2, 1):
                    end = i + width
                    if not self.available(i, end):
                        continue
                    value = key(self.phrase(i, end))
                    state = value if codes_only and value in codes else self.lex['states'].get(value) if not codes_only else None
                    if not state:
                        continue
                    if value == 'fl' and i and (NUMBER.fullmatch(self.words[i-1]) or ORDINAL.fullmatch(self.words[i-1])):
                        continue
                    if end < len(self.words) and self.words[end] not in SEPARATORS and not ZIP.fullmatch(self.words[end]):
                        postal_marker = any(
                            self.phrase(end, end+width) in self.lex['postal_markers']
                            for width in (2, 1) if end+width <= len(self.words)
                        )
                        # Reverse order without commas: Boston MA 123 Main St.
                        numbered_tail = NUMBER.fullmatch(self.words[end]) and any(w in self.lex['street'] for w in self.words[end+1:end+self.limit])
                        tail = self.words[end:end+self.limit]
                        tail = tail[:next((j for j,w in enumerate(tail) if w in SEPARATORS),len(tail))]
                        leading_state = i == 0 and any(NUMBER.fullmatch(w) for w in tail[:6]) and any(w in self.lex['street'] for w in tail)
                        if not postal_marker and not numbered_tail and not leading_state:
                            continue
                    if not codes_only and 'state' in self.c.values and value in city_index()[0]:
                        continue
                    # CT can be a street suffix; IN/OR can be ordinary words.
                    if width == 1 and value in {'ct', 'in', 'or', 'me', 'hi', 'id'} and i > 0 and self.words[i-1] not in SEPARATORS:
                        prefix = self.words[max(0, i-self.limit):i]
                        if not any(w in self.lex['street'] for w in prefix) and any(NUMBER.fullmatch(w) for w in prefix):
                            continue
                    self.emit(i, end, 'STATE', {'state': state})

    def floors(self):
        for i, word in enumerate(self.words):
            for width in (2, 1):
                end = i+width
                phrase = self.phrase(i, end)
                if phrase in self.lex['floor_special']:
                    self.emit(i, end, 'FLOOR', {'floor': self.lex['floor_special'][phrase]},
                              'floor', [('FLOOR_SPECIAL', phrase)])
            if i+1 >= len(self.words):
                continue
            following = self.words[i+1]
            if word in self.lex['floor'] and (NUMBER.fullmatch(following) or ORDINAL.fullmatch(following)):
                number, typed = following, [('FLOOR_KEY', word), ('ORDINAL' if ORDINAL.fullmatch(following) else 'NUMBER', following)]
            elif following in self.lex['floor'] and (NUMBER.fullmatch(word) or ORDINAL.fullmatch(word)):
                number, typed = word, [('ORDINAL' if ORDINAL.fullmatch(word) else 'NUMBER', word), ('FLOOR_KEY', following)]
            else:
                continue
            ordinal = ORDINAL.fullmatch(number)
            self.emit(i, i+2, 'FLOOR', {'floor': 'floor '+(ordinal[1] if ordinal else number)}, 'floor', typed)

    def units_and_boxes(self):
        for i, word in enumerate(self.words):
            if key(word) == 'po' and i+2 < len(self.words) and self.words[i+1] == 'box':
                value = self.words[i+2]
                if IDENTIFIER.fullmatch(value):
                    self.emit(i, i+3, 'POBOX', {'po_box': 'po box '+value}, 'po_box', [('BOX_KEY','po box'), ('ID', value)])
            if word not in self.lex['unit']:
                continue
            end = i
            while end < min(i+3, len(self.words)) and self.words[end] in self.lex['unit']:
                end += 1
            if end >= len(self.words) or not IDENTIFIER.fullmatch(self.words[end]):
                continue
            # Dataset uses #8340 Burds Rd as a house number. A suffix #5 is a unit.
            if word == '#' and end+2 < len(self.words):
                ahead = self.words[end+1:min(end+self.limit, len(self.words))]
                before_separator = ahead[:next((j for j,w in enumerate(ahead) if w in SEPARATORS), len(ahead))]
                if any(w in self.lex['street'] for w in before_separator):
                    continue
            markers = [self.lex['unit'][w] for w in self.words[i:end]]
            value = ' '.join([*markers, self.words[end]])
            self.emit(i, end+1, 'UNIT', {'unit': value}, 'unit',
                      [('UNIT_KEY', w) for w in markers]+[('ID', self.words[end])])

    def landmarks(self):
        for i in range(len(self.words)):
            for width in (3, 2, 1):
                end = i+width
                if self.phrase(i,end) not in self.lex['landmark'] or not self.available(i,end):
                    continue
                finish = end
                while finish < min(len(self.words), i+self.limit) and self.available(finish,finish+1):
                    w = self.words[finish]
                    if NUMBER.fullmatch(w) or w in self.lex['unit'] or w in self.lex['floor']:
                        break
                    finish += 1
                if finish > end:
                    self.emit(i,finish,'LANDMARK', {'landmark_direction': self.phrase(i,finish)},
                              'landmark', [('LANDMARK_KEY',self.phrase(i,end))]+[('WORD',w) for w in self.words[end:finish]])
                break

    @staticmethod
    def house_token(word: str) -> bool:
        return bool(IDENTIFIER.fullmatch(word) and any(c.isdigit() for c in word) and not ORDINAL.fullmatch(word))

    def streets(self):
        for i, word in enumerate(self.words):
            if word not in self.lex['street'] or self.c.claimed[i]:
                continue
            start = i
            while start > 0 and i-start < self.limit-1 and self.available(start-1, i+1):
                if (self.words[start-1] in self.lex['building'] and self.words[start-1] not in self.lex['street']) or self.words[start-1] in self.lex['unit']:
                    break
                start -= 1
            # Prefer a numbered street suffix over a preceding St (= Saint).
            house = None
            house_start = start
            for j in range(start, i):
                if self.house_token(self.words[j]):
                    house_start, house = j, self.words[j]
                    break
            if house is not None:
                start = house_start
                name_start = start+1
                # 172-77 A Highland Ave: A is an address-number suffix.
                if name_start+1 < i and len(self.words[name_start]) == 1 and self.words[name_start] not in self.lex['direction']:
                    house += ' '+self.words[name_start]
                    name_start += 1
            else:
                name_start = start
            end = i+1
            route = word in {'route','rte','highway','hwy'} and end < len(self.words) and NUMBER.fullmatch(self.words[end]) and self.available(start,end+1)
            if route:
                end += 1
                typed = ([('HOUSE',house)] if house else [])+[('WORD',w) for w in self.words[name_start:i]]+[('STREET_KEY',word), ('NUMBER',self.words[i+1])]
                street = ' '.join([*self.words[name_start:i],self.lex['street'][word],self.words[i+1]])
            else:
                if name_start >= i:
                    continue
                # A leading "St" belongs to St Charles Ave, not to a street named 123.
                typed = ([('HOUSE',house)] if house else [])+[('WORD',w) for w in self.words[name_start:i]]+[('STREET_KEY',word)]
                street = ' '.join([*(self.lex['direction'].get(w,w) for w in self.words[name_start:i]), self.lex['street'][word]])
                if end < len(self.words) and self.words[end] in self.lex['direction'] and self.available(start,end+1):
                    typed.append(('DIRECTION',self.words[end]))
                    street += ' '+self.lex['direction'][self.words[end]]
                    end += 1
            values = {'street': street}
            if house:
                values['house_number'] = house
                if start > 0 and self.words[start-1] == '#' and self.available(start-1,end):
                    start -= 1
                    values['house_number'] = '#'+house
            self.emit(start,end,'STREET',values,'street',typed)

    def postcodes(self):
        for i, word in enumerate(self.words):
            if not ZIP.fullmatch(word):
                continue
            previous_state = any(s.kind == 'STATE' and s.end in (i,i-1) for s in self.c.spans)
            terminal = i+1 == len(self.words) or self.words[i+1] in SEPARATORS
            marker_start = i
            marker = None
            for width in (2, 1):
                start = i-width
                if start >= 0 and self.available(start, i+1):
                    candidate = self.phrase(start, i)
                    if candidate in self.lex['postal_markers']:
                        marker_start, marker = start, candidate
                        break
            if previous_state or marker or terminal and ('street' in self.c.values or 'po_box' in self.c.values):
                typed = ([('POSTAL_KEY', marker)] if marker else []) + [('POSTAL', word)]
                self.emit(marker_start,i+1,'POSTAL',{'postal_code': word},'postal_code',typed)

    def buildings_and_localities(self):
        city_names = city_index()[0]
        for start,end in list(self.chunks()):
            if end-start > self.limit:
                continue
            phrase = self.phrase(start,end)
            # City evidence wins over a building/locality suffix (e.g. Beverly Hills).
            if key(phrase) in city_names:
                continue
            for field, kind in [('building','BUILDING'), ('locality','LOCALITY')]:
                if phrase in self.lex['known_'+('buildings' if field == 'building' else 'localities')]:
                    typed = [('KNOWN_'+kind,phrase)]
                elif end-start > 1 and self.words[end-1] in self.lex[field]:
                    typed = [('WORD',w) for w in self.words[start:end-1]]+[(kind+'_KEY',self.words[end-1])]
                elif end-start > 1 and self.words[start] in self.lex[field]:
                    typed = [(kind+'_KEY',self.words[start])]+[('WORD',w) for w in self.words[start+1:end]]
                else:
                    continue
                if self.emit(start,end,kind,{field:phrase},field,typed):
                    break

    def cities(self, *, known_only: bool = False):
        index, regional = city_index()
        state = self.c.values.get('state')
        chunks = list(self.chunks())
        for start,end in chunks:
            for city_start in range(max(start,end-5),end):
                # A city may follow a marked building/locality without a comma.
                if city_start != start and not any(w in self.lex['building'] or w in self.lex['locality'] for w in self.words[start:city_start]):
                    continue
                candidate = key(self.phrase(city_start,end))
                hits = {city for city, expected in index.get(candidate,[]) if not state or expected == state}
                if (len(candidate.replace(' ','')) <= 3 or candidate in regional) and not state:
                    continue
                if len(hits) == 1:
                    self.emit(city_start,end,'CITY',{'city':hits.pop()})
                    break
        if known_only or 'city' in self.c.values or not state or not ({'street','po_box'} & self.c.values.keys()):
            return
        # Unknown cities need a short complete segment adjacent to a state,
        # or the final segment after a recognized street. Never scan substrings.
        candidates = []
        state_spans = [s for s in self.c.spans if s.kind == 'STATE']
        for start,end in self.chunks():
            words = self.words[start:end]
            if end-start > 5 or any(not w.replace('-','').replace("'",'').isalpha() for w in words):
                continue
            if any(w in self.lex['missing_markers'] or w in self.lex['unit'] or w in self.lex['landmark'] or w in self.lex['building'] or w in self.lex['floor'] for w in words):
                continue
            distance = min(min(abs(start-s.end),abs(end-s.start)) for s in state_spans)
            final_city = end == len(self.words) and any(s.kind == 'STREET' and s.end < start for s in self.c.spans)
            if distance <= 1 or final_city:
                candidates.append((distance,start,end))
        if candidates:
            _,start,end = min(candidates)
            self.emit(start,end,'CITY',{'city':self.phrase(start,end)})

    def result(self) -> AddressResult:
        self.states()
        self.units_and_boxes()
        self.floors()
        self.landmarks()
        self.streets()
        self.postcodes()
        self.cities(known_only=True)
        self.buildings_and_localities()
        self.cities()
        values = self.c.values
        remainder = self.c.remainder()
        assembled = accepts('address',[(s.kind,self.phrase(s.start,s.end)) for s in sorted(self.c.spans,key=lambda s:s.start)]) if self.c.spans else False
        complete = bool((values.get('street') and values.get('house_number') or values.get('po_box')) and values.get('city') and values.get('state'))
        status = 'parsed' if complete and assembled and not remainder else 'partial' if values else 'fallback'
        reasons = []
        if self.c.conflicts:
            reasons.append('conflicting_'+','.join(sorted(self.c.conflicts)))
        if remainder:
            reasons.append('unmatched_text')
        if values and not complete:
            reasons.append('incomplete_components')
        if values and not assembled:
            reasons.append('unrecognized_order')
        return AddressResult(**values,unparsed_text=remainder,parser='us_regex_lark',parse_status=status,
                             parse_reason=';'.join(reasons) or None)


def _parse_address(text: str | None) -> AddressResult:
    if not text or not text.strip():
        return AddressResult(parser='us_regex_lark',parse_status='empty')
    return _Parser(text).result()


def parse_address(text: str | None) -> AddressResult:
    return _parse_address(text)

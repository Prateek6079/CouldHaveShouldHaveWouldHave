"""India extraction with a granular-to-large order prior and lossless leftovers."""
from functools import lru_cache
import json
from pathlib import Path
import re

from .address_schema import AddressResult
from .address_normalization import normalize_india_address
from .component_grammar import Span, SpanCollector, accepts

NUMBER = re.compile(r'\d+(?:[-/]\d+[a-z0-9]*)*\Z')
ORDINAL = re.compile(r'\d+(?:st|nd|rd|th)\Z')
IDENTIFIER = re.compile(r'[a-z0-9]+(?:[-/][a-z0-9]+)*\Z')
PIN = re.compile(r'[1-9]\d{5}\Z')
SEPARATORS = {',', ';', '(', ')'}


def _key(text: str) -> str:
    """Fold lexical punctuation for lookup without rewriting source text."""
    text = text.lower().replace("'", '').replace('’', '').replace('.', '')
    text = text.replace('&', ' ').replace('-', ' ')
    return ' '.join(text.split())


@lru_cache(maxsize=1)
def lexicon() -> dict:
    path = Path(__file__).resolve().parents[1] / 'resources/india_address_lexicon.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('version') != 1 or data.get('country') != 'india':
        raise ValueError('Unsupported India address lexicon version')
    return data


@lru_cache(maxsize=1)
def _indexes():
    data = lexicon()
    state_index: dict[str, set[tuple[str, bool]]] = {}
    state_width = 1
    for aliases, abbreviated in ((data['states'], False), (data['state_abbreviations'], True)):
        for alias, canonical in aliases.items():
            normalized_alias = _key(alias)
            state_width = max(state_width, len(normalized_alias.split()))
            state_index.setdefault(normalized_alias, set()).add((canonical, abbreviated))
    city_index: dict[str, str] = {}
    for alias, canonical in data.get('city_aliases', {}).items():
        city_index[_key(alias)] = canonical

    return state_index, state_width, city_index


@lru_cache(maxsize=128)
def _alias_index(aliases):
    return {_key(alias): alias for alias in aliases}


class _Parser:
    def __init__(self, text: str):
        self.c = SpanCollector(text)
        self.words = [token.text for token in self.c.tokens]
        self.lex = lexicon()
        self.limit = self.lex['max_component_tokens']
        self.state_index, self.state_width, self.city_index = _indexes()
        self.location_markers = set(self.lex['street']) | set(self.lex['floor']) | set(self.lex['unit'])
        self.location_markers |= set(self.lex['landmark']) | set(self.lex['building'])
        self.location_markers |= {word for aliases in self.lex['administrative_areas'].values()
                                  for alias in aliases for word in _key(alias).split()}
        self.location_markers |= {word for aliases in self.lex['number_markers'].values()
                                  for alias in aliases for word in _key(alias).split()}

    def phrase(self, start: int, end: int) -> str:
        return ' '.join(self.words[start:end])

    def available(self, start: int, end: int) -> bool:
        return (0 <= start < end <= len(self.words) and end-start <= self.limit
                and self.c.free(start, end)
                and not any(word in SEPARATORS for word in self.words[start:end]))

    def emit(self, start: int, end: int, kind: str, values: dict,
             rule: str | None = None, typed: list | None = None) -> bool:
        if not self.available(start, end):
            return False
        if rule and not accepts(rule, typed or [], country='india'):
            return False
        if 'locality' in values and 'locality' in self.c.values:
            del self.c.values['locality']
            added = self.c.add(Span(start, end, kind, values))
            self.c.values['locality'] = ', '.join(
                span.values['locality'] for span in sorted(self.c.spans, key=lambda s: s.start)
                if 'locality' in span.values)
            return added
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

    @staticmethod
    def _variants(phrase: str) -> set[str]:
        # normalized_expr(address=True) preserves punctuation. These variants
        # are lookup-only and cover the tokenizer's trailing-period behavior.
        base = ' '.join(phrase.lower().split())
        return {base, _key(base)}

    def _phrase_match(self, start: int, end: int, aliases) -> tuple[str, str] | None:
        key = _key(self.phrase(start, end))
        alias = _alias_index(tuple(aliases)).get(key)
        return (alias, key) if alias is not None else None

    def countries(self):
        for i, word in enumerate(self.words):
            tail = [w for w in self.words[i+1:] if w not in SEPARATORS]
            if word == 'india' and (not tail or PIN.fullmatch(tail[0])):
                self.emit(i, i+1, 'COUNTRY', {})

    def states(self):
        matches = []
        for start in range(len(self.words)):
            for end in range(min(len(self.words), start+self.state_width, start+self.limit), start, -1):
                if not self.available(start, end):
                    continue
                entries = set()
                for variant in self._variants(self.phrase(start, end)):
                    entries.update(self.state_index.get(variant, set()))
                if not entries:
                    continue
                canonical = {state for state, _ in entries}
                abbreviated = any(flag for _, flag in entries)
                if len(canonical) != 1:
                    continue
                before = start == 0 or self.words[start-1] in SEPARATORS
                after = end == len(self.words) or self.words[end] in SEPARATORS
                next_word = end
                while next_word < len(self.words) and self.words[next_word] in SEPARATORS:
                    next_word += 1
                pin_nearby = next_word < len(self.words) and bool(PIN.fullmatch(self.words[next_word]))
                marker_nearby = any(
                    self.phrase(max(0, start-width), start) in self.lex['postal_markers']
                    for width in (1, 2) if start >= width
                )
                numeric_or_road_context = any(NUMBER.fullmatch(w) for w in self.words[:start]) or any(
                    w in self.lex['street'] or w in self.lex['main_cross_markers']
                    for w in self.words[:start]
                )
                tail = [w for j, w in enumerate(self.words[end:], end) if w not in SEPARATORS and not self.c.claimed[j]]
                if tail and not (before and after or PIN.fullmatch(tail[0]) or tail[0] in self.lex['postal_markers']):
                    continue
                if abbreviated:
                    if not (before and after or pin_nearby or marker_nearby or after and numeric_or_road_context):
                        continue
                elif not (before or after or pin_nearby or marker_nearby):
                    continue
                matches.append((start, end, next(iter(canonical))))
        if matches:
            def score(item):
                start, end, _ = item
                next_word = end
                while next_word < len(self.words) and self.words[next_word] in SEPARATORS:
                    next_word += 1
                followed_by_pin = next_word < len(self.words) and bool(PIN.fullmatch(self.words[next_word]))
                boundary_after = end == len(self.words) or self.words[end] in SEPARATORS
                return (followed_by_pin, boundary_after, end, end-start)
            # Prefer the location-state occurrence nearest a PIN/end. This also
            # lets a repeated state name remain available as a city candidate.
            start, end, canonical = max(matches, key=score)
            self.emit(start, end, 'STATE', {'state': canonical})

    def postcodes(self):
        markers = self.lex['postal_markers']
        for i, word in enumerate(self.words):
            if not PIN.fullmatch(word) or self.c.claimed[i]:
                continue
            marker_start, marker = i, None
            for width in (3, 2, 1):
                start = i-width
                if start >= 0 and self.available(start, i):
                    match = self._phrase_match(start, i, markers)
                    if match:
                        marker_start, marker = start, match[0]
                        break
            previous_state = any(span.kind in {'STATE', 'COUNTRY'} and span.end in (i, i-1) for span in self.c.spans)
            terminal = i+1 == len(self.words) or self.words[i+1] in SEPARATORS
            address_marker_context = any(w in self.lex['street'] or w in self.lex['main_cross_markers']
                                          or w in self.lex['locality'] for w in self.words[:i])
            address_marker_context |= any(
                _key(' '.join(self.words[max(0, i-width):i])) == _key(marker_text)
                for marker_text in self.lex['po_box_markers']
                for width in range(1, min(4, i)+1)
            )
            city_context = any(_key(self.phrase(start, i)) in self.city_index
                               for start in range(max(0, i-3), i))
            if marker or previous_state or terminal and (address_marker_context or city_context):
                typed = ([('POSTAL_KEY', marker)] if marker else []) + [('PIN', word)]
                self.emit(marker_start, i+1, 'POSTAL', {'postal_code': word}, 'postal_code', typed)

    def units_and_boxes(self):
        for i in range(len(self.words)):
            # PO boxes and post bags are India-recognized delivery identifiers.
            for end in range(min(len(self.words), i+4), i, -1):
                if not self.available(i, end):
                    continue
                marker = self._phrase_match(i, end, self.lex['po_box_markers'])
                if marker and end < len(self.words) and IDENTIFIER.fullmatch(self.words[end]):
                    value = 'post bag ' + self.words[end] if marker[1] == 'post bag' else 'po box ' + self.words[end]
                    self.emit(i, end+1, 'POBOX', {'po_box': value}, 'po_box',
                              [('BOX_KEY', marker[1]), ('ID', self.words[end])])
                    break

        marker_groups = [
            ('unit_number', 'UNIT'), ('plot_number', 'HOUSE'),
            ('survey_number', 'HOUSE'), ('house_number', 'HOUSE'),
        ]
        for group, kind in marker_groups:
            aliases = self.lex['number_markers'][group]
            for i in range(len(self.words)):
                for end in range(min(len(self.words), i+4), i, -1):
                    if not self.available(i, end) or end >= len(self.words):
                        continue
                    match = self._phrase_match(i, end, aliases)
                    identifier = self.words[end]
                    if not match:
                        continue
                    if match[1] in {'no', 'number', '#'} and i and self.words[i-1] in self.location_markers:
                        continue
                    if identifier in {'no', 'number'} and end+1 < len(self.words) and IDENTIFIER.fullmatch(self.words[end+1]):
                        end += 1
                        identifier = self.words[end]
                    if not self.is_identifier(identifier):
                        continue
                    if kind == 'UNIT':
                        base = match[1].split()[0]
                        value = f'{base} {identifier}'
                        typed = [('UNIT_KEY', base), ('ID', identifier)]
                        self.emit(i, end+1, kind, {'unit': value}, 'unit', typed)
                    else:
                        category = ('plot' if group == 'plot_number' else
                                    'survey' if group == 'survey_number' else 'house')
                        value = f'{category} {identifier}' if category in {'plot', 'survey'} else identifier
                        self.emit(i, end+1, kind, {'house_number': value}, 'house_no',
                                  [('HOUSE_KEY', category), ('ID', identifier)])
                    break

        unit_aliases = self.lex['unit']
        for i, word in enumerate(self.words):
            if word not in unit_aliases or i+1 >= len(self.words) or not self.available(i, i+2):
                continue
            end = i+1
            if self.words[end] in {'no', 'number'} and end+1 < len(self.words):
                end += 1
            identifier = self.words[end]
            if not self.is_identifier(identifier):
                continue
            canonical = unit_aliases[word]
            self.emit(i, end+1, 'UNIT', {'unit': f'{canonical} {identifier}'}, 'unit',
                      [('UNIT_KEY', canonical), ('ID', identifier)])

    @staticmethod
    def is_identifier(word):
        return bool(IDENTIFIER.fullmatch(word) and
                    (any(c.isdigit() for c in word) or len(word) == 1))

    def bare_identifier(self):
        # An unlabelled leading number is a unit when followed by a floor,
        # otherwise a premise number. Alphanumeric IDs remain valid.
        if not self.words or self.c.claimed[0] or not self.is_identifier(self.words[0]):
            return
        word = self.words[0]
        if not any(c.isdigit() for c in word) or ORDINAL.fullmatch(word) or PIN.fullmatch(word):
            return
        if len(self.words) > 1 and self.words[1] in self.lex['locality']:
            return
        if (len(self.words) > 1 and self.words[1] in self.lex['main_cross_markers']
                and not (self.words[1] == 'main' and len(self.words) > 2 and self.words[2] == 'road')):
            return
        following = sorted((span for span in self.c.spans if span.start > 0), key=lambda s: s.start)
        kind = 'UNIT' if following and following[0].kind == 'FLOOR' and 'unit' not in self.c.values else 'HOUSE'
        if 'house_number' in self.c.values and 'unit' not in self.c.values:
            kind = 'UNIT'
        field = 'unit' if kind == 'UNIT' else 'house_number'
        self.emit(0, 1, kind, {field: word}, 'unit' if kind == 'UNIT' else 'house_no', [('ID', word)])

    def floors(self):
        for i in range(len(self.words)):
            for width in (3, 2, 1):
                end = i+width
                if end > len(self.words) or not self.available(i, end):
                    continue
                phrase = self.phrase(i, end)
                special = next((v for k, v in self.lex['floor_special'].items() if _key(k) == _key(phrase)), None)
                if special:
                    self.emit(i, end, 'FLOOR', {'floor': special}, 'floor', [('FLOOR_SPECIAL', special)])
            if i+1 >= len(self.words) or not self.available(i, i+2):
                continue
            word, following = self.words[i], self.words[i+1]
            if word in self.lex['floor'] and (NUMBER.fullmatch(following) or ORDINAL.fullmatch(following)):
                number, key = following, word
                typed = [('FLOOR_KEY', key), ('ORDINAL' if ORDINAL.fullmatch(number) else 'NUMBER', number)]
            elif following in self.lex['floor'] and (NUMBER.fullmatch(word) or ORDINAL.fullmatch(word)):
                number, key = word, following
                typed = [('ORDINAL' if ORDINAL.fullmatch(number) else 'NUMBER', number), ('FLOOR_KEY', key)]
            else:
                continue
            ordinal = ORDINAL.fullmatch(number)
            self.emit(i, i+2, 'FLOOR', {'floor': 'floor '+(re.match(r'\d+', number)[0] if ordinal else number)}, 'floor', typed)

    def landmarks(self):
        aliases = self.lex['landmark']
        for i in range(len(self.words)):
            for end in range(min(len(self.words), i+4), i, -1):
                if not self.available(i, end):
                    continue
                match = self._phrase_match(i, end, aliases)
                if not match:
                    continue
                finish = end
                while finish < min(len(self.words), i+7) and self.available(finish, finish+1):
                    word = self.words[finish]
                    if (NUMBER.fullmatch(word) or word in self.lex['floor']):
                        break
                    finish += 1
                if finish > end:
                    relation = aliases[match[0]]
                    tail = self.words[end:finish]
                    typed = [('LANDMARK_KEY', relation)] + [('WORD', w) for w in tail]
                    self.emit(i, finish, 'LANDMARK', {'landmark_direction': ' '.join([relation, *tail]),
                                                       }, 'landmark', typed)
                break

    def _marker_match(self, words: list[str], markers) -> tuple[int, int, str] | None:
        index = _alias_index(tuple(markers))
        for start in range(len(words)):
            for end in range(min(len(words), start+4), start, -1):
                marker = index.get(_key(' '.join(words[start:end])))
                if marker is not None:
                    return start, end, marker
        return None

    def buildings_localities(self, *, buildings):
        field, kind = ('building', 'BUILDING') if buildings else ('locality', 'LOCALITY')
        markers = self.lex[field]
        for start, end in list(self.chunks()):
            known = self.lex['known_buildings' if buildings else 'known_localities']
            if self._phrase_match(start, end, known):
                self.emit(start, end, kind, {field: self.phrase(start, end)})
                continue
            # Peel multiple named/numbered components from a delimiter-free run.
            while start < end:
                words = self.words[start:end]
                marker = self._marker_match(words, markers)
                if not marker:
                    break
                left, right, matched = marker
                if not buildings and any(w in self.location_markers - set(markers) for w in words[:left]):
                    break
                finish = start+right
                if left == 0:
                    if right == len(words):
                        break
                    if self.is_identifier(words[right]):
                        finish += 1
                    else:
                        finish = end
                    typed = [(kind+'_KEY', _key(matched))] + [('WORD', w) for w in self.words[start+right:finish]]
                else:
                    # A building suffix may be followed by its number: Tower 2.
                    if right < len(words) and self.is_identifier(words[right]):
                        finish += 1
                    # Do not turn "Nagar Main Road" into a locality/building.
                    if buildings and any(w in self.lex['street'] for w in words[:left]):
                        break
                    typed = [('WORD', w) for w in words[:left]] + [(kind+'_KEY', _key(matched))]
                    typed += [('WORD', w) for w in self.words[start+right:finish]]
                if not self.emit(start, finish, kind, {field: self.phrase(start, finish)}, field, typed):
                    break
                start = finish

    def administrative(self):
        areas = self.lex['administrative_areas']
        for start, end in list(self.chunks()):
            words = self.words[start:end]
            matched = None
            for group in ('village', 'post_office', 'subdistrict', 'mandal', 'district', 'block'):
                match = self._marker_match(words, areas[group])
                if match:
                    matched = (group, match)
                    break
            if not matched:
                continue
            group, (left, right, marker) = matched
            if left != 0 and right != len(words):
                continue
            if group == 'block':  # Urban Block A is a building marker.
                continue
            name_words = words[right:] if left == 0 else words[:left]
            if not name_words:
                continue
            if group == 'village':
                kind, field = 'VILLAGE', 'city'
                rule = 'village_component'
                key_type = 'VILLAGE_KEY'
            elif group == 'district':
                kind, field = 'DISTRICT', 'locality'
                rule = 'district'
                key_type = 'DISTRICT_KEY'
            elif group == 'post_office':
                kind, field = 'POST_OFFICE', 'locality'
                rule, key_type = None, None
            elif group == 'subdistrict':
                kind, field = 'SUBDISTRICT', 'locality'
                rule, key_type = None, None
            elif group == 'mandal':
                kind, field = 'MANDAL', 'locality'
                rule, key_type = None, None
            else:
                kind, field = 'BLOCK', 'locality'
                rule, key_type = None, None
            if rule:
                typed = ([(key_type, _key(marker))] + [('WORD', w) for w in name_words]
                         if left == 0 else [('WORD', w) for w in name_words] + [(key_type, _key(marker))])
            else:
                typed = None
            value = ' '.join(name_words) if field == 'city' else self.phrase(start, end)
            self.emit(start, end, kind, {field: value}, rule, typed)

    def streets(self):
        marker_map = {**dict.fromkeys(self.lex['main_cross_markers']), **self.lex['street']}
        marker_map = {_key(k): v or k for k, v in marker_map.items()}
        routes = {'nh', 'national highway', 'sh', 'state highway', 'highway', 'gali', 'galli', 'lane', 'road', 'street'}
        for start, end in list(self.chunks()):
            matches = []
            i = start
            while i < end:
                match = None
                for stop in range(min(end, i+3), i, -1):
                    key = _key(self.phrase(i, stop))
                    if key in marker_map:
                        match = (i, stop, key, marker_map[key])
                        break
                if match:
                    matches.append(match)
                    i = match[1]
                else:
                    i += 1
            if not matches:
                continue
            left, finish, key, canonical = matches[-1]
            # Prefix route/lane marker followed by a number.
            if left == start and key in routes and finish < end:
                if self.words[finish] in {'no', 'number'} and finish+1 < end:
                    finish += 1
                if not self.is_identifier(self.words[finish]):
                    continue
                identifier = self.words[finish]
                self.emit(start, finish+1, 'STREET', {'street': f'{canonical} {identifier}'},
                          'street', [('STREET_KEY', canonical), ('NUMBER', identifier)])
                continue
            if left == start and key == 'main road':
                self.emit(start, finish, 'STREET', {'street': canonical}, 'street', [('STREET_KEY', canonical)])
                continue
            if left == start:
                # A bare "Road" is insufficient to identify a street.
                continue
            prefix = self.words[start:left]
            if (len(prefix) > 1 and self.is_identifier(prefix[0]) and any(c.isdigit() for c in prefix[0])
                    and not ORDINAL.fullmatch(prefix[0])):
                if self.emit(start, start+1, 'HOUSE', {'house_number': prefix[0]}, 'house_no', [('ID', prefix[0])]):
                    start += 1
                    prefix = prefix[1:]
            typed = [('WORD', w) for w in prefix] + [('STREET_KEY', canonical)]
            self.emit(start, finish, 'STREET', {'street': ' '.join([*prefix, canonical])}, 'street', typed)

    def cities(self, *, infer=False):
        if 'city' in self.c.values:
            return
        anchors = [s for s in self.c.spans if s.kind in {'STATE', 'POSTAL'}]
        boundary = min((s.start for s in anchors), default=len(self.words))
        short = {_key(a) for a in self.lex.get('short_city_aliases', [])}
        matches = []
        for start, end in self.chunks():
            # Known cities can be a suffix of an undelimited address. Prefer the
            # rightmost, longest alias and never match inside a road name.
            for left in range(start, end):
                alias = _key(self.phrase(left, end))
                canonical = self.city_index.get(alias)
                if end > boundary and left != start:
                    continue
                if canonical and (alias not in short or anchors and left == start):
                    if left != start and not (anchors or any(c.isdigit() for c in self.phrase(start, left))):
                        continue
                    matches.append((end, end-left, left, canonical))
        if matches:
            end, _, start, canonical = max(matches)
            self.emit(start, end, 'CITY', {'city': canonical})
            return
        # Delhi etc. can be both a city and a state, but a separately written
        # city (e.g. Chandigarh, Punjab) takes precedence over this sharing.
        for span in anchors:
            state = span.values.get('state')
            if state in self.city_index.values():
                if span.start and self.words[span.start-1] == 'new' and state == 'delhi':
                    self.emit(span.start-1, span.start, 'CITY', {'city': state})
                else:
                    self.c.values['city'] = state
                return
        if not infer or not anchors:
            return
        delivery = [s for s in self.c.spans if s.kind in {'STREET', 'HOUSE', 'UNIT', 'POBOX', 'BUILDING'}]
        blocked = set(self.lex['missing_markers']) | set(self.lex['unit']) | set(self.lex['floor'])
        blocked |= set(self.lex['building']) | set(self.lex['locality']) | set(self.lex['street'])
        blocked |= self.location_markers
        for start, end in reversed(list(self.chunks())):
            words = self.words[start:end]
            if not (0 <= boundary-end <= 1 and 1 <= len(words) <= 4 and any(s.end <= start for s in delivery)):
                continue
            if any(w in blocked or not w.replace('-', '').replace("'", '').isalpha() for w in words):
                continue
            if len(''.join(words)) >= 3:
                self.emit(start, end, 'CITY', {'city': self.phrase(start, end)})
                return

    def ordered_localities(self):
        # Only infer a name between a delivery component and a city. Names
        # before the delivery or after the state remain explicitly unparsed.
        city = [s for s in self.c.spans if s.kind in {'CITY', 'VILLAGE'}]
        if not city and self.c.values.get('city') == self.c.values.get('state'):
            city = [s for s in self.c.spans if s.kind == 'STATE']
        delivery = [s for s in self.c.spans if s.kind in {'STREET', 'HOUSE', 'UNIT', 'BUILDING'}]
        if not city or not delivery:
            return
        low, high = max(s.end for s in delivery), min(s.start for s in city)
        blocked = set(self.lex['missing_markers']) | self.location_markers | set(self.lex['locality'])
        for start, end in list(self.chunks()):
            words = self.words[start:end]
            phrase = _key(self.phrase(start, end))
            if phrase in self.city_index or phrase in self.state_index:
                continue
            if low <= start < end <= high and len(words) <= 6 and all(
                w.replace('-', '').replace("'", '').isalpha() and w not in blocked for w in words
            ):
                self.emit(start, end, 'LOCALITY', {'locality': self.phrase(start, end)})

    def result(self) -> AddressResult:
        self.countries()
        self.states()
        self.postcodes()
        self.units_and_boxes()
        self.floors()
        self.cities()
        self.landmarks()
        self.bare_identifier()
        self.buildings_localities(buildings=True)
        self.streets()
        self.administrative()
        self.buildings_localities(buildings=False)
        self.cities(infer=True)
        self.ordered_localities()
        values = self.c.values
        remainder = self.c.remainder()
        ordered = sorted(self.c.spans, key=lambda span: span.start)
        assembled = accepts('address', [(span.kind, self.phrase(span.start, span.end)) for span in ordered if span.kind not in {'LANDMARK', 'COUNTRY'}], country='india') if ordered else False
        complete = bool(values.get('state') and values.get('city') and any(values.get(f) for f in ('street', 'po_box', 'house_number', 'unit', 'building')))
        status = 'parsed' if complete and assembled and not remainder and not self.c.conflicts else 'partial' if values else 'fallback'
        reasons = []
        if self.c.conflicts:
            reasons.append('conflicting_'+','.join(sorted(self.c.conflicts)))
        if remainder:
            reasons.append('unmatched_text')
        if values and not complete:
            reasons.append('incomplete_components')
        if values and not assembled:
            reasons.append('unrecognized_order')
        return AddressResult(**values, unparsed_text=remainder, parser='india_regex_lark',
                             parse_status=status, parse_reason=';'.join(reasons) or None)


def parse_address(text: str | None) -> AddressResult:
    if not text or not text.strip():
        return AddressResult(parser='india_regex_lark', parse_status='empty')
    return _Parser(normalize_india_address(text)).result()

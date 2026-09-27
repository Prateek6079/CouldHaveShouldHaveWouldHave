"""Specification examples, ambiguity boundaries and partial-result contracts."""
import unittest
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal

from src.preprocessing.address import enrich_address_batch
from src.preprocessing.component_grammar import accepts
from src.preprocessing.us import lexicon, parse_address


class USGrammarTests(unittest.TestCase):
    def check(self, text, **expected):
        result = parse_address(text)
        for field, value in expected.items():
            self.assertEqual(getattr(result,field), value, (text,field,result))
        return result

    def test_attached_complete_examples(self):
        self.check('Apt 7B, 350 W 42nd St, Manhattan, New York, NY',
                   house_number='350', street='w 42nd st', unit='apt 7b',
                   locality='manhattan', city='new york city', state='ny', parse_status='parsed')
        self.check('500 Park Avenue Suite 400 New York NY', house_number='500',
                   street='park ave',unit='suite 400', city='new york city', state='ny', unparsed_text=None)

    def test_partial_example_retains_unknown(self):
        self.check('Suite 420, Jefferson Tower, Somewhere near Central Park',
                   unit='suite 420', building='jefferson tower', landmark_direction='near central park',
                   unparsed_text='Somewhere', city=None, street=None, parse_status='partial')
        self.check('123 Main St, Boston',house_number='123',street='main st',city='boston',state=None,parse_status='partial')
        self.check('unrecognized strange text',unparsed_text='unrecognized strange text',parse_status='fallback')

    def test_unit_identifiers(self):
        for marker in lexicon()['unit']:
            for identifier in ['4','4B','B4','12-A','A-12','PH1']:
                with self.subTest(marker=marker, identifier=identifier):
                    self.check(f'{marker} {identifier}',unit=f"{lexicon()['unit'][marker]} {identifier.lower()}",unparsed_text=None)
        self.check('600 9th Street, Unit Lot 23, Bismarck, ND',unit='unit lot 23',street='9th st',city='bismarck')

    def test_floors(self):
        for text,expected in [('3rd Floor','floor 3'),('Floor 3','floor 3'),('Level 10','floor 10'),
                              ('10th Fl.','floor 10'),('3 Floor','floor 3'),('Floor 3rd','floor 3'),
                              ('Ground level','ground floor'),('Basement','basement'),('Lower level','lower level'),
                              ('Mezzanine','mezzanine'),('Penthouse','penthouse')]:
            self.check(text,floor=expected,unit=None,unparsed_text=None)

    def test_buildings_and_localities(self):
        for text in ['Empire State Building','One Liberty Plaza','Tower B','Jefferson Center','Parkview Apartments','Union Station']:
            self.check(text,building=text.lower(),unparsed_text=None)
        for text in ['Oakwood Estates','Lincoln Park','Beverly Hills','Battery Park','Greenwood Village','Neighborhood Oakwood']:
            self.check(text,locality=text.lower(),unparsed_text=None)

    def test_streets_and_numbers(self):
        for house in ['123','221B','12-14','18/20','00123','W14698']:
            self.check(f'{house} Main Street',house_number=house.lower(),street='main st',postal_code=None)
        for text,street in [('123 North Main Street','n main st'),('123 W 42nd Street','w 42nd st'),
                            ('123 Main Street NW','main st nw'),('123 St Charles Ave','st charles ave'),
                            ('409 Court Street','court st'),('123 Highway 10','hwy 10'),('7196 State Route 14','state rte 14')]:
            self.check(text,street=street,unparsed_text=None)
        self.check('#8340 BURDS RD, PEOSTA TOWNSHIP, IA',house_number='#8340',unit=None,city='peosta township')
        self.check('172-77 A Highland Avenue, Jamaica, NY',house_number='172-77 a',street='highland ave')

    def test_known_city_after_building_without_comma(self):
        self.check('123 Main St Empire Tower Boston MA',building='empire tower',city='boston',unparsed_text=None)
        self.check('LA, 266 Enterprise Drive, Houma',street='enterprise dr',city='houma',state='la')

    def test_states_and_postcode(self):
        for name,code in lexicon()['states'].items():
            for state in (name,code):
                self.check(f'123 Main St, Sampleville, {state} 00123',state=code,city='sampleville',postal_code='00123')
        self.check('12345 Main St, Boston, MA 02108-0001',house_number='12345',postal_code='02108-0001')
        self.check('PO Box 004, Boston MA 02108',po_box='po box 004',postal_code='02108',house_number=None)
        for marker in lexicon()['postal_markers']:
            with self.subTest(marker=marker):
                self.check(f'123 Main St, Boston, MA, {marker} 02108',
                           postal_code='02108',unparsed_text=None,parse_status='parsed')
                self.check(f'123 Main St, Boston, MA {marker} 02108',
                           state='ma',postal_code='02108',unparsed_text=None,parse_status='parsed')
        self.check('123 Washington St, Boston, MA',street='washington st',state='ma',city='boston')
        self.check('123 Main Ct, Hartford, CT',street='main ct',state='ct',city='hartford')

    def test_order_and_building_composition(self):
        for text in ['123 Main St, Boston, MA','Boston, MA, 123 Main St','MA, Boston, 123 Main St',
                     'Boston, 123 Main St, MA','123 Main St, MA, Boston',
                     'Boston MA 123 Main St','MA Boston 123 Main St',
                     'Apt 4B, 123 Main St, Boston, MA','123 Main St Apt 4B, Boston, MA',
                     'Suite 500, 5th Floor, Empire Tower, 123 Main St, Boston, MA',
                     '123 Main St, Empire Tower, 5th Floor, Suite 500, Boston, MA']:
            self.check(text,house_number='123',street='main st',city='boston',state='ma',unparsed_text=None,parse_status='parsed')

    def test_landmarks(self):
        for relation in lexicon()['landmark']:
            for text in [f'{relation} City Hall, 123 Main St, Boston, MA',f'123 Main St, Boston, MA, {relation} City Hall']:
                self.check(text,landmark_direction=f'{relation} city hall',street='main st',city='boston',unparsed_text=None,parse_status='parsed')
        self.check('Near Main Street, 123 Park Avenue, Boston, MA',landmark_direction='near main street',street='park ave')

    def test_conflicting_duplicate_components_keep_evidence(self):
        result=self.check('Apt 4, Apt 9, 123 Main St, Boston, MA',unit='apt 4',unparsed_text='Apt 9',parse_status='partial')
        self.assertIn('conflicting_unit',result.parse_reason)
        self.check('Apt 4, Apt 4',unit='apt 4',unparsed_text='Apt 4')
        self.check('123 Main St, NULL, Harrisville, West Virginia',city='harrisville',state='wv',unparsed_text='NULL')

    def test_name_bound_and_no_model_fallback(self):
        text=' '.join(['Mystery']*40)
        self.check(text,unparsed_text=text,parse_status='fallback')
        with patch('usaddress.tag',side_effect=AssertionError('default parser invoked the old model')):
            self.check('123 Main St, Boston, MA',city='boston')

    def test_grammar_rejects_bad_shapes(self):
        self.assertFalse(accepts('unit',[('ID','4')]))
        self.assertFalse(accepts('address',[('CITY','boston'),('UNIT','apt 4'),('STATE','ma')]))
        self.assertTrue(accepts('floor',[('ORDINAL','3rd'),('FLOOR_KEY','floor')]))

    def test_real_address_regressions(self):
        import json
        from pathlib import Path
        fixture=json.loads((Path(__file__).parent/'fixtures/us_address_regressions.json').read_text())
        for text,*values in fixture['cases']:
            self.check(text,**dict(zip(('house_number','street','city','state'),values)))

    def test_batch_preserves_input_and_schema(self):
        batch=pl.DataFrame({'entity_id':['x','y'], 'business_address':['123 Main St, Boston, MA',None], 'country':['US','US']})
        before=batch.clone()
        result=enrich_address_batch(batch)
        assert_frame_equal(batch,before)
        assert_frame_equal(result,enrich_address_batch(result))
        self.assertEqual(result['business_address'].to_list(),batch['business_address'].to_list())
        self.assertEqual(result['address_parser'].to_list(),['us_regex_lark']*2)
        self.assertEqual(result['address_parse_status'].to_list(),['parsed','empty'])

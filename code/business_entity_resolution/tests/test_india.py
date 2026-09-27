"""India parsing regressions: semantic assignments, ordering and retention."""
from collections import Counter
import re
import unittest

import polars as pl
from polars.testing import assert_frame_equal

from src.preprocessing import preprocess_batch
from src.preprocessing.address import parse_address, enrich_address_batch
from src.preprocessing.address_normalization import normalize_india_address
from src.preprocessing.address_schema import FIELD_NAMES


class IndiaTests(unittest.TestCase):
    def check(self, text, **expected):
        result = parse_address(text, 'India')
        for name, value in expected.items():
            self.assertEqual(getattr(result, name), value, (text, name, result))
        return result

    def test_reported_roman_floor_and_slash_identifier(self):
        self.check('17/18, III Floor S K C Buildings, Mill Road, Coimbatore, Tamil Nadu',
                   unit='17/18', floor='floor 3', building='s k c buildings', street='mill road',
                   city='coimbatore', state='tamil nadu', unparsed_text=None, parse_status='parsed')

    def test_reported_leading_number_and_locality(self):
        self.check('303, Krishna Puri Main Road, Mandawali, Delhi', house_number='303',
                   street='krishna puri main road', locality='mandawali', city='delhi',
                   state='delhi', unparsed_text=None, parse_status='parsed')

    def test_reported_plot_and_unmarked_locality(self):
        self.check('Plot No. 20B, Gaur Bheet Faizullaganj, Lucknow, Uttar Pradesh',
                   house_number='plot 20b', locality='gaur bheet faizullaganj',
                   city='lucknow', unparsed_text=None, parse_status='parsed')

    def test_contextual_roman_normalization(self):
        for raw, expected in [('III Floor', '3 Floor'), ('Floor IX', 'Floor 9'),
                              ('IV Fl.', '4 Fl.'), ('Phase II', 'Phase 2'),
                              ('XII Sector', '12 Sector'), ('MMXX Floor', '2020 Floor')]:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_india_address(raw), expected)
                self.assertEqual(normalize_india_address(expected), expected)
        for name in ['S K C Buildings', 'IV Labs', 'Civil Lines', 'IIV Floor', 'IIII Floor', 'Mix Road']:
            self.assertEqual(normalize_india_address(name), name)

    def test_floor_variants(self):
        for raw, value in [('III Floor', 'floor 3'), ('Floor IV', 'floor 4'),
                           ('10th Fl.', 'floor 10'), ('3rd Floor', 'floor 3'),
                           ('Lower Ground Floor', 'lower ground'), ('Upper Ground Floor', 'upper ground')]:
            self.check(raw, floor=value, unparsed_text=None)

    def test_unit_labels_are_numeric_or_identifier_not_names(self):
        for text in ['Unit No. 12/3', 'Flat No 12/3', 'Apartment No. 12/3', 'Shop 12/3']:
            result = self.check(text, unparsed_text=None)
            self.assertIn('12/3', result.unit)
        self.check('Apartment Green, Mumbai, Maharashtra', unit=None)
        self.check('Office Road, Mumbai, Maharashtra', unit=None, street='office road')
        self.check('Unit No Lotus, Mumbai, Maharashtra', unit=None)

    def test_bare_numbers_and_alphanumeric_ids(self):
        for identifier in ['303', '17/18', '12/A', 'A-12', '16-11-23/37/A']:
            self.check(f'{identifier}, MG Road, Bangalore, Karnataka', house_number=identifier.lower(), unparsed_text=None)
        self.check('303, 3rd Floor, MG Road, Bangalore, Karnataka', unit='303', house_number=None)
        self.check('H.No.16-11-23/37/A, MG Road, Hyderabad, Telangana', house_number='16-11-23/37/a', unparsed_text=None)

    def test_numbered_streets_are_not_houses(self):
        self.check('3 Main, Bangalore, Karnataka', street='3 main', house_number=None)
        self.check('3rd Main Road, Bangalore, Karnataka', street='3rd main road', house_number=None)
        self.check('5th Main 2nd Cross Road, Bangalore, Karnataka', street='5th main 2nd cross', unparsed_text=None)
        self.check('NH 44, Hyderabad, Telangana', street='national highway 44', unparsed_text=None)

    def test_locality_markers_at_either_end(self):
        for locality in ['Anna Nagar', 'Sector 12', '12 Sector', 'Phase II', 'II Phase', 'Rajiv Gandhi Nagar']:
            result = self.check(f'12, MG Road, {locality}, Chennai, Tamil Nadu', unparsed_text=None)
            self.assertEqual(result.locality, normalize_india_address(locality).lower())

    def test_leading_numbered_locality_is_not_a_house(self):
        self.check('12 Sector, Noida, Uttar Pradesh', house_number=None, locality='12 sector', unparsed_text=None, parse_status='partial')
        self.check('II Phase, Noida, Uttar Pradesh', house_number=None, locality='2 phase', unparsed_text=None, parse_status='partial')

    def test_multiple_localities_and_number_retention(self):
        self.check('12, MG Road, Anna Nagar, Sector 12, Chennai, Tamil Nadu',
                   locality='anna nagar, sector 12', unparsed_text=None, parse_status='parsed')

    def test_city_lookup_without_delivery_or_state(self):
        for text, city in [('Mumbai, Maharashtra', 'mumbai'), ('Bengaluru', 'bangalore'),
                           ('Navi Mumbai, Maharashtra', 'navi mumbai'), ('New Delhi', 'delhi'),
                           ('BLR, Karnataka', 'bangalore')]:
            self.check(text, city=city, unparsed_text=None)
        self.check('BLR', city=None)

    def test_city_suffix_without_commas(self):
        self.check('12 MG Road Bangalore Karnataka 560001', house_number='12', street='mg road',
                   city='bangalore', state='karnataka', postal_code='560001', unparsed_text=None)
        self.check('12 MG Road Navi Mumbai Maharashtra', city='navi mumbai', street='mg road', unparsed_text=None)

    def test_city_state_overlap_and_repeated_name(self):
        self.check('12, MG Road, Delhi, Delhi', city='delhi', state='delhi', unparsed_text=None)
        self.check('12, MG Road, New Delhi, Delhi', city='delhi', state='delhi', unparsed_text=None)
        self.check('12, MG Road, Chandigarh, Punjab', city='chandigarh', state='punjab', unparsed_text=None)

    def test_city_and_state_names_inside_roads(self):
        self.check('12 Mumbai Road, Pune, Maharashtra', street='mumbai road', city='pune', state='maharashtra')
        self.check('12 Delhi Road, Mumbai, Maharashtra', street='delhi road', city='mumbai', state='maharashtra')
        self.check('12 Karnataka Road', street='karnataka road', state=None, city=None)
        self.check('12 Delhi Road', street='delhi road', state=None, city=None)

    def test_unknown_city_inference_respects_order(self):
        self.check('12, MG Road, Smalltown, Karnataka', city='smalltown', unparsed_text=None)
        self.check('Smalltown, Karnataka', city=None, unparsed_text='Smalltown')
        self.check('12, MG Road, Karnataka, Acme', city=None)

    def test_reversed_order_is_partial_and_keeps_fields(self):
        result = self.check('Bangalore, Karnataka, 12 MG Road', parse_status='partial', city='bangalore', state='karnataka')
        self.assertNotEqual(result.parse_status, 'parsed')
        self.assertTrue(result.street)

    def test_administrative_middle_marker_does_not_drop_numbers(self):
        text = '12, MG Road, Foo Village 26, Bangalore, Karnataka'
        result = self.check(text)
        self.assertIn('26', result.unparsed_text or result.locality or '')

    def test_conflicting_units_are_retained(self):
        result = self.check('Flat 12, Flat 13, MG Road, Mumbai, Maharashtra', unit='flat 12', parse_status='partial')
        self.assertIn('13', result.unparsed_text)
        self.assertIn('conflicting_unit', result.parse_reason)

    def test_pins_and_invalid_pin(self):
        for text in ['12 MG Road, Bangalore, Karnataka 560001', 'Bangalore 560001',
                     'PIN Code 560001', 'PO Box 12, Bangalore, Karnataka 560001']:
            self.check(text, postal_code='560001', unparsed_text=None)
        self.check('12 MG Road, Karnataka 000000', postal_code=None)
        self.check('Unit 123456, Mumbai, Maharashtra', unit='unit 123456', postal_code=None)

    def test_attached_and_spaced_pins_need_location_context(self):
        for text, postal in [('Chennai600004, Tamil Nadu', '600004'),
                             ('12 MG Road, Pune411014, Maharashtra', '411014'),
                             ('12 MG Road, Madurai-625 001, Tamil Nadu', '625001'),
                             ('12 MG Road, Bangalore, Karnataka, India560001', '560001'),
                             ('PIN Code 560 001', '560001')]:
            self.check(text, postal_code=postal, unparsed_text=None)
        for text in ['Flat C030103, Pune, Maharashtra', 'Unit A102103, Surat, Gujarat',
                     '155-156, MG Road, Delhi', 'U70200TN2024OPC168918, Chennai, Tamil Nadu']:
            self.check(text, postal_code=None)

    def test_concatenated_structural_markers(self):
        cases = {
            '18 r n mukherjee rd6th floor': '18 r n mukherjee rd 6th floor',
            '6thfloor, nanjappabuilding': '6th floor, nanjappa building',
            'jlnmarg, takteshairoad': 'jln marg, takteshai road',
            'govindihouseplotno8-a': 'govindihouse plot no 8-a',
            'kanotabaghbehindrbi': 'kanotabagh behind rbi',
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_india_address(raw), expected)
                self.assertEqual(normalize_india_address(expected), expected)

    def test_concatenation_normalization_is_conservative(self):
        for value in ['warehouse', 'abroad', 'roadside', 'margaret road',
                      'building materials', 'flooring shop', 'flatiron building']:
            with self.subTest(value=value):
                self.assertEqual(normalize_india_address(value), value)

    def test_concatenated_failure_examples_parse_further(self):
        result = self.check(
            '18 r n mukherjee rd6th floor, kolkata, calcutta, west bengal',
            floor='floor 6', city='kolkata', state='west bengal')
        self.assertNotIn('rd6th', result.unparsed_text or '')
        result = self.check(
            '17, first floor, nanjappabuilding, kalyanagar, t dasarahalli, '
            'bangalore, karnataka',
            building='nanjappa building', city='bangalore', state='karnataka')
        self.assertNotIn('nanjappabuilding', result.unparsed_text or '')

    def test_building_and_locality_delimiter_free(self):
        self.check('Flat 12 III Floor Lotus Apartments MG Road Anna Nagar Chennai Tamil Nadu',
                   unit='flat 12', floor='floor 3', building='lotus apartments', street='mg road',
                   locality='anna nagar', city='chennai', unparsed_text=None, parse_status='parsed')
        self.check('12, MG Road, Sector 5 Phase II, Bangalore, Karnataka', locality='sector 5, phase 2', unparsed_text=None)

    def test_numerics_never_disappear(self):
        for text in ['17/18, III Floor S K C Buildings, Mill Road, Coimbatore, Tamil Nadu',
                     '12, MG Road, Foo Village 26, Bangalore, Karnataka',
                     'Flat 12, Flat 13, MG Road, Mumbai, Maharashtra',
                     '12, Lake Town Block A 26, Kolkata, West Bengal',
                     'Survey No 17/18, Village Foo, Taluk Bar 26, Karnataka']:
            result = parse_address(text, 'India')
            output = ' '.join(getattr(result, f) or '' for f in FIELD_NAMES[:FIELD_NAMES.index('parser')])
            self.assertFalse(Counter(re.findall(r'\d+', text)) - Counter(re.findall(r'\d+', output)), (text, result))

    def test_production_normalization_and_country_scope(self):
        batch = pl.DataFrame({'entity_id': ['1', '2', '3'], 'business_name': ['IV Labs']*3,
                              'business_address': ['III Floor, Mumbai, Maharashtra']*3,
                              'country': ['India', 'US', None]})
        before = batch.clone()
        normalized = preprocess_batch(batch)
        self.assertEqual(normalized['business_address'].to_list(),
                         ['3 floor, mumbai, maharashtra', 'iii floor, mumbai, maharashtra', 'iii floor, mumbai, maharashtra'])
        result = enrich_address_batch(normalized)
        self.assertEqual(result['address_floor'][0], 'floor 3')
        assert_frame_equal(batch, before)
        assert_frame_equal(preprocess_batch(normalized), normalized)

    def test_number_marker_and_numbered_road(self):
        self.check('No. 4, Bangalore, Karnataka', house_number='4', unparsed_text=None)
        self.check('#328, Chennai, TN', house_number='328', unparsed_text=None)
        self.check('Road No 1, Hyderabad, Telangana', street='road 1', house_number=None, unparsed_text=None)
        self.check('103, Sai Towers D.No. 3-6-290, Hyderabad, Telangana',
                   unit='103', house_number='3-6-290', building='sai towers', unparsed_text=None)

    def test_unparsed_component_is_not_relabelled_as_locality(self):
        result = self.check('12, First Road, Second Road, Bangalore, Karnataka', locality=None)
        self.assertIn('Second Road', result.unparsed_text)
        self.check('12, MG Road, Mumbai, Mumbai, Maharashtra', locality=None, unparsed_text='Mumbai')

    def test_empty_unknown_and_partial_contract(self):
        self.check(None, parse_status='empty')
        self.check('Unrecognized Words', parse_status='fallback', unparsed_text='Unrecognized Words')
        self.check('Floor 3', floor='floor 3', parse_status='partial', unparsed_text=None)


if __name__ == '__main__':
    unittest.main()

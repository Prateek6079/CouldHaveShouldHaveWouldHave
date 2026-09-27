import hashlib
from pathlib import Path
import tempfile
import unittest
from src.preprocessing.us_legacy import parse_address as parse_legacy_us
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal
import usaddress

from src.data_io import read_source
from src.preprocessing import preprocess_batches
from src.preprocessing.address import parse_address, enrich_address_batch, enrich_address_batches
from src.preprocessing.address_schema import ADDRESS_COLUMNS


class AddressTests(unittest.TestCase):
    def test_us_street_and_missing_postcode(self):
        result=parse_address('1795 westchester drive, high point, nc','US')
        self.assertEqual((result.house_number,result.street,result.city,result.state),
                         ('1795','westchester dr','high point','nc'))
        self.assertIsNone(result.postal_code)
        self.assertEqual(result.parse_status,'parsed')

    def test_us_unit_zip_and_po_box(self):
        result=parse_address('123 main st suite 004, boston ma 02108-0001',' USA ')
        self.assertEqual(result.unit,'suite 004')
        self.assertEqual(result.postal_code,'02108-0001')
        result=parse_address('po box 004, boston ma 02108','United States')
        self.assertEqual(result.po_box,'po box 004')
        self.assertIsNone(result.house_number)

    def test_us_unassigned_and_normalization(self):
        tags={'AddressNumber':'12','StreetNamePreDirectional':'North','StreetName':'Main',
              'StreetNamePostType':'Street','PlaceName':'Boston','StateName':'MA',
              'ZipCode':'not-a-zip','Recipient':'Shop ABC'}
        with patch('src.preprocessing.us_legacy.usaddress.tag',return_value=(tags,'Street Address')):
            result=parse_legacy_us('input')
        self.assertEqual(result.street,'n main st')
        self.assertIsNone(result.postal_code)
        self.assertIn('shop abc',result.unparsed_text)
        self.assertIn('not-a-zip',result.unparsed_text)
        self.assertEqual(result.parse_status,'partial')

    def test_us_ambiguous_parse_falls_back(self):
        error=usaddress.RepeatedLabelError('input', [('one','StreetName'),('two','StreetName')], 'StreetName')
        with patch('src.preprocessing.us_legacy.usaddress.tag',side_effect=error):
            result=parse_legacy_us('ambiguous input')
        self.assertEqual(result.unparsed_text,'ambiguous input')
        self.assertEqual(result.parse_status,'fallback')
        self.assertEqual(result.parse_reason,'RepeatedLabelError')

    def test_french_rules_and_leading_zero(self):
        result=parse_address('14 bis rue victor hugo, 01000 bourg-en-bresse','FR')
        self.assertEqual((result.house_number,result.street,result.postal_code,result.city),
                         ('14 bis','rue victor hugo','01000','bourg-en-bresse'))
        self.assertEqual(result.parse_status,'partial')
        self.assertEqual(result.parser,'fr_rules')

    def test_french_unknown_text_retained(self):
        result=parse_address('batiment b, 75002 paris cedex 01','France')
        self.assertEqual(result.city,'paris')
        self.assertIn('batiment b',result.unparsed_text)
        self.assertIn('cedex 01',result.unparsed_text)
        result=parse_address('75001 ou 75002','France')
        self.assertIsNone(result.postal_code)
        self.assertEqual(result.parse_status,'fallback')

    def test_empty_unknown_and_india_routing(self):
        for country in ['US','France','India',None,'unknown']:
            self.assertEqual(parse_address(None,country).parse_status,'empty')
        result=parse_address('12 main road','India')
        self.assertEqual(result.parse_status,'partial')
        self.assertIsNone(result.unparsed_text)
        self.assertEqual(result.house_number,'12')
        self.assertEqual(result.street,'main road')
        self.assertEqual(parse_address('abc','Mars').parse_reason,'unsupported_country')

    def test_batch_schema_order_read_only_and_repeat(self):
        batch=pl.DataFrame({'entity_id':['S1-002','S1-001'],
                            'business_address':['75002 paris','12 road'], 'country':['France','India']})
        before=batch.clone()
        result=enrich_address_batch(batch)
        assert_frame_equal(batch,before)
        assert_frame_equal(result,enrich_address_batch(result))
        self.assertEqual(result['entity_id'].to_list(),['S1-002','S1-001'])
        self.assertEqual(result['business_address'].to_list(),batch['business_address'].to_list())
        self.assertEqual(set(result.columns),set(batch.columns)|set(ADDRESS_COLUMNS))
        self.assertTrue(all(result.schema[c]==pl.String for c in ADDRESS_COLUMNS))
        self.assertEqual(enrich_address_batch(batch.head(0)).height,0)

    def test_stream_input_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.tsv'
            p.write_text('entity_id\tbusiness_name\tbusiness_address\tcountry\n'
                         'S1-01\tCafé SARL\t12 rue de la Paix, 75002 Paris\tFrance\n',encoding='utf-8')
            before=(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns)
            batches=list(enrich_address_batches(preprocess_batches(read_source(p,'S1',batch_size=1))))
            self.assertEqual(batches[0]['address_city'][0],'paris')
            self.assertEqual(before,(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns))

    def test_arguments(self):
        with self.assertRaises(ValueError): enrich_address_batch(pl.DataFrame())

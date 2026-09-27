import unittest
from src.preprocessing.us_legacy import parse_address as parse_legacy_us
from unittest.mock import patch
import polars as pl
from src.preprocessing.address import parse_address, enrich_address_batch
from src.preprocessing.address_normalization import normalize_component, normalize_french_street


class AddressNormalizationTests(unittest.TestCase):
    def test_us_guessed_roles_are_used_without_reinterpreting_name(self):
        tags={'AddressNumber':'123','StreetName':'St Charles','StreetNamePostType':'Avenue',
              'PlaceName':'Boston','StateName':'MA'}
        with patch('src.preprocessing.us_legacy.usaddress.tag',return_value=(tags,'Street Address')):
            self.assertEqual(parse_legacy_us('input').street,'st charles ave')
        # Trust a type label even when another interpretation could exist.
        tags['StreetNamePostType']='St.'
        with patch('src.preprocessing.us_legacy.usaddress.tag',return_value=(tags,'Street Address')):
            self.assertEqual(parse_legacy_us('input').street,'st charles st')

    def test_us_equivalent_types_and_unknown(self):
        for long, short in [('Avenue','ave'),('Road','rd'),('Street','st')]:
            self.assertEqual(normalize_component(long,'us','street_types'),short)
            self.assertEqual(normalize_component(short+'.','us','street_types'),short)
        self.assertEqual(normalize_component('Unlisted Way','us','street_types'),'unlisted way')

    def test_french_abbreviations_and_names(self):
        for alias, canonical in [('av.','avenue'),('bd','boulevard'),('r.','rue'),('rte','route'),('chem.','chemin')]:
            with self.subTest(alias=alias):
                result=parse_address(f'12 {alias} st charles, 75002 paris','France')
                self.assertEqual(result.street,f'{canonical} st charles')
                self.assertEqual(result.city,'paris')
                self.assertEqual(normalize_french_street(result.street),result.street)
        self.assertEqual(normalize_french_street('saint charles'),'saint charles')

    def test_whole_text_status_and_unparsed_text_preserved(self):
        batch=pl.DataFrame({'business_address':['12 av. st charles, batiment b, 75002 paris'],'country':['France']})
        result=enrich_address_batch(batch)
        self.assertEqual(result['business_address'][0],batch['business_address'][0])
        self.assertEqual(result['address_street'][0],'avenue st charles')
        self.assertIn('batiment b',result['address_unparsed_text'][0])
        self.assertEqual(result['address_parse_status'][0],'partial')

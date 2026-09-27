import unittest
from src.preprocessing.cities import enrich_us_city, enrich_french_city, city_index
from src.preprocessing.address_schema import AddressResult
from src.preprocessing.address import parse_address


class CityTests(unittest.TestCase):
    def test_supplied_aliases(self):
        for name,state,expected in [('NYC','ny','new york city'),('L.A.','ca','los angeles'),
                ('Ft. Worth','tx','fort worth'),("B'more",'md','baltimore'),('Saint Louis','mo','st. louis')]:
            self.assertEqual(enrich_us_city(AddressResult(city=name,state=state),'input').city,expected)

    def test_recover_reordered_city_without_changing_evidence(self):
        original=AddressResult(unparsed_text='philly, pa',parse_status='partial')
        result=enrich_us_city(original,'philly, pa, 123 main st')
        self.assertEqual(result.city,'philadelphia')
        self.assertEqual(result.unparsed_text,original.unparsed_text)
        self.assertEqual(result.parse_status,original.parse_status)

    def test_conflicts_short_aliases_and_street_names(self):
        for text in ['123 austin street','la','dfw']:
            self.assertIsNone(enrich_us_city(AddressResult(),text).city)
        self.assertEqual(enrich_us_city(AddressResult(city='portland',state='me'),'portland me').city,'portland')
        self.assertIsNone(enrich_us_city(AddressResult(state='me'),'portland, me').city)
        self.assertIsNone(enrich_us_city(AddressResult(),'austin, dallas, tx').city)

    def test_french_aliases_and_missing_postcode(self):
        for name,expected in [('Marseilles','marseille'),('Nîmes','nimes'),('St Etienne','saint-etienne'),
                              ('Straßburg','strasbourg'),('Mülhausen','mulhouse')]:
            self.assertEqual(enrich_french_city(AddressResult(),f'12 rue principale, {name}').city,expected)
        self.assertEqual(parse_address('105 quai des chevillards, lille, hauts-de-france','France').city,'lille')
        self.assertEqual(enrich_french_city(AddressResult(),'12 rue x, 75002 paris cedex 01').city,'paris')

    def test_reunion_requires_context(self):
        self.assertIsNone(enrich_french_city(AddressResult(),'12 rue x, st-denis').city)
        self.assertEqual(enrich_french_city(AddressResult(),'12 rue x, st-denis, reunion').city,'saint-denis (reunion)')
        self.assertEqual(enrich_french_city(AddressResult(),'12 rue x, 97400 saint-denis').city,'saint-denis (reunion)')

    def test_country_scope_and_existing_unknown_city(self):
        self.assertIsNone(enrich_french_city(AddressResult(),'123 rue paris').city)
        self.assertIsNone(enrich_french_city(AddressResult(),'boston').city)
        self.assertEqual(enrich_french_city(AddressResult(city='unlisted town'),'paris').city,'unlisted town')
        self.assertIsNone(parse_address('paris','India').city)
        self.assertIsNone(parse_address('paris','unknown').city)

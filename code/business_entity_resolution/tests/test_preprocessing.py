import hashlib
from pathlib import Path
import tempfile
import unittest

import polars as pl
from polars.testing import assert_frame_equal

from src.data_io import read_source
from src.preprocessing import preprocess_batch, preprocess_batches
from src.preprocessing.transliteration import romanize, UnsupportedScriptError


def frame(names, addresses=None, countries=None):
    size = len(names)
    return pl.DataFrame({'entity_id': [f'S1-{i:03}' for i in range(size)],
                         'business_name': names, 'business_address': addresses or [''] * size,
                         'country': countries or ['US'] * size},
                        schema={k: pl.String for k in ('entity_id','business_name','business_address','country')})


class PreprocessingTests(unittest.TestCase):
    def test_case_spacing_accents_and_compatibility(self):
        result = preprocess_batch(frame(['  CAFÉ\tServices  ', 'Ｃａｆｅ\u0301', 'Œuvre & Sons', "O’Reilly's"]))
        self.assertEqual(result['business_name'].to_list(), ['cafe services','cafe','oeuvre and sons','oreillys'])

    def test_legal_boundaries(self):
        names = ['ACME Pvt. Ltd.', 'LLC Acme', 'Acme L.L.C.', 'Limited Edition',
                 'The Limited Store', 'Lincoln', 'LLC', 'Acme Co', 'Acme Ltd LLC']
        result = preprocess_batch(frame(names))['business_name'].to_list()
        self.assertEqual(result, ['acme','acme','acme','edition','the limited store','lincoln','llc','acme co','acme'])

    def test_country_legal_forms(self):
        names = ['SARL Café SAS', 'SARL Café SAS', 'SA Boutique', 'SARL', 'SARL Acme Ltd SAS']
        countries = ['France','US','FR','France','France']
        self.assertEqual(preprocess_batch(frame(names,countries=countries))['business_name'].to_list(),
                         ['cafe','sarl cafe sas','boutique','sarl','acme'])

    def test_address_structure_and_digits(self):
        addresses = ['१२/३ Main–Road, Apt #००४', '12-14 rue de l’Église, 75001', 'Bât. A / 02']
        self.assertEqual(preprocess_batch(frame(['A']*3,addresses))['business_address'].to_list(),
                         ['12/3 main-road, apt #004', "12-14 rue de l'eglise, 75001", 'bat. a / 02'])

    def test_nine_indic_scripts_and_mixed_text(self):
        for value in ['भारत','বাংলা','ਪੰਜਾਬ','ગુજરાત','ଓଡ଼ିଆ','தமிழ்','తెలుగు','ಕನ್ನಡ','മലയാളം']:
            with self.subTest(value=value):
                result = romanize(value)
                self.assertTrue(result)
                self.assertTrue(result.isascii(), result)
        self.assertEqual(preprocess_batch(frame(['श्री साई Services Ltd']))['business_name'][0], 'shri sai services')

    def test_unsupported_script_fails(self):
        with self.assertRaisesRegex(UnsupportedScriptError, 'business_name'):
            preprocess_batch(frame(['Москва']))

    def test_input_unchanged_no_extra_columns_and_idempotence(self):
        source = frame(['Café LLC','श्री साई Ltd'], ['12/3 rue École','१२ Main Road'])
        before = source.clone()
        result = preprocess_batch(source)
        assert_frame_equal(source, before)
        self.assertEqual(source.columns, result.columns)
        self.assertEqual(source['entity_id'].to_list(), result['entity_id'].to_list())
        assert_frame_equal(result, preprocess_batch(result))

    def test_empty_null_and_empty_frame(self):
        result = preprocess_batch(frame([None,'','   ']))
        self.assertEqual(result['business_name'].to_list(), ['','',''])
        self.assertEqual(preprocess_batch(frame([])).height, 0)

    def test_invalid_schema(self):
        with self.assertRaises(ValueError):
            preprocess_batch(pl.DataFrame({'business_name':['a']}))
        with self.assertRaises(ValueError):
            preprocess_batch(frame(['a']).with_columns(pl.lit(3).alias('business_name')))

    def test_batch_stream_and_read_only_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.tsv'
            path.write_text('entity_id\tbusiness_name\tbusiness_address\tcountry\n'
                            'S1-001\tCafé LLC\t12/3 Road\tUS\n'
                            'S1-002\tश्री साई Ltd\t१२ Main Road\tIndia\n', encoding='utf-8')
            def stamp():
                return hashlib.sha256(path.read_bytes()).hexdigest(),path.stat().st_mtime_ns
            before = stamp()
            batches = list(preprocess_batches(read_source(path,'S1',batch_size=1)))
            self.assertEqual([b.height for b in batches], [1,1])
            self.assertEqual(pl.concat(batches)['business_name'].to_list(), ['cafe','shri sai'])
            self.assertEqual(stamp(),before)


    def test_modern_script_characters(self):
        for value in ['സൂപ്പർ ഫിനാൻസ്', 'ഓൾ', 'കൺ', 'ഓട്ടോമൊബൈൽ', 'ସ୍ୱସ୍ତିକ', 'ऑटो', 'मॉडर्न']:
            with self.subTest(value=value):
                self.assertTrue(romanize(value).isascii())
        self.assertEqual(romanize('മോഡേൺ'), 'moden')


    def test_transliterated_legal_forms(self):
        result = preprocess_batch(frame(['श्री साई प्राइवेट लिमिटेड'], countries=['India']))
        self.assertEqual(result['business_name'][0], 'shri sai')

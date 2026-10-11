# SPDX-License-Identifier: MIT

import json
from pathlib import Path
import tempfile
import unittest

import testing_audit as audit


class TestingAuditTest(unittest.TestCase):
    def test_generic_reference_does_not_hide_missing_family_test(self):
        result = audit.report({'goc_a', 'goc_b'}, {
            'goc_a': {'instruction_flags_test.cpp'},
            'goc_b': {'integer_test.cpp', 'api_test.cpp'},
        }, {})
        self.assertEqual(result['missing_family_references'], ['goc_a'])
        self.assertEqual(result['unreviewed_family_files'], ['integer_test.cpp'])
        self.assertEqual(result['instructions'][1]['reviewed_test_files'], [])

    def test_only_explicit_reviews_are_attached(self):
        review = {'file': 'integer_test.cpp', 'scope': 'one specific domain'}
        result = audit.report({'goc_a'}, {'goc_a': {'integer_test.cpp'}},
                              {'integer_test.cpp': review})
        self.assertEqual(result['instructions'][0]['reviewed_test_files'], ['integer_test.cpp'])
        self.assertEqual(result['reviews'], [review])
        self.assertIn('not a per-mode proof', result['limitations'])

    def test_compilation_database_excludes_stale_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'source'
            build = Path(tmp) / 'build'
            (root / 'tests').mkdir(parents=True)
            build.mkdir()
            (build / 'active.o').touch()
            (build / 'stale.o').touch()
            entry = {'file': str(root / 'tests/active_test.cpp'), 'directory': str(build),
                     'arguments': ['c++', '-c', 'active_test.cpp', '-o', 'active.o']}
            (build / 'compile_commands.json').write_text(json.dumps([entry]))
            self.assertEqual(audit.active_objects(build, root),
                             {build / 'active.o': 'active_test.cpp'})
            (build / 'active.o').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing test object'):
                audit.active_objects(build, root)

    def test_review_checks_assertion_and_evidence_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'tests').mkdir()
            (root / 'tests/example_test.cpp').write_text('TEST_P(Example, Exact) {}')
            (root / 'tests/golden.h').touch()
            path = root / 'tests/reference_reviews.json'
            record = {'file': 'example_test.cpp', 'assertions': ['Example.Exact'],
                      'evidence': ['tests/golden.h'], 'scope': 'a captured domain'}
            path.write_text(json.dumps([record]))
            self.assertEqual(list(audit.read_reviews(root)), ['example_test.cpp'])
            record['assertions'] = ['Example.Missing']
            path.write_text(json.dumps([record]))
            with self.assertRaisesRegex(ValueError, 'Missing reviewed test assertion'):
                audit.read_reviews(root)
            record['assertions'] = ['Example.Exact']
            record['evidence'] = ['tests/missing.h']
            path.write_text(json.dumps([record]))
            with self.assertRaisesRegex(ValueError, 'Missing oracle evidence'):
                audit.read_reviews(root)


if __name__ == '__main__':
    unittest.main()

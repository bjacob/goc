#!/usr/bin/env python3
"""Tests for review-queue extraction, not GPU semantic equivalence."""
import unittest
import json
from pathlib import Path
import tempfile
import sys

sys.dont_write_bytecode = True
from architecture_audit import direct_policies, execution_functions, read_scope, read_reviews


class ExtractionTest(unittest.TestCase):
    def test_nested_lambdas_comments_and_strings_do_not_hide_following_helper(self):
        source = '''
// inline void execute_fake() { }
template <typename Inst>
inline void execute_v_first(Inst &inst) {
  auto nested = [] { return "} fake {"; };
  /* } */ fp_mode::effective_omod(inst.arch(), 0, false, 1);
}
inline void execute_v_second() { fp_mode::quiets_nan(0, true); }
'''
        functions = list(execution_functions(source))
        self.assertEqual([f[0] for f in functions], ['execute_v_first', 'execute_v_second'])
        self.assertIn('OMOD', direct_policies(functions[0][1]))
        self.assertNotIn('NaN quieting', direct_policies(functions[0][1]))
        self.assertIn('NaN quieting', direct_policies(functions[1][1]))

    def test_comments_do_not_create_architecture_evidence(self):
        [(name, body, _)] = execution_functions('''inline void execute_v_same() {
          // fp_mode::effective_omod(wf.arch());
          auto text = "fp_mode::quiets_nan";
          ROCJITSU_TRY_SIMD_FOO();
        }''')
        self.assertEqual(name, 'execute_v_same')
        self.assertEqual(direct_policies(body), {'SIMD macro'})

    def test_truncated_input_fails_instead_of_silently_dropping_evidence(self):
        with self.assertRaises(ValueError):
            list(execution_functions('inline void execute_broken() {'))

    def test_scope_keeps_architecture_exclusive_and_boundary_rows(self):
        text = '''| `v_a` | yes | yes | in | x | — | pending |
| `v_b` | — | yes | in: register routing | x | — | pending |
| `s_c` | yes | yes | out: scheduling | — | — | excluded |'''
        self.assertEqual(read_scope(text), [('v_a', True, True), ('v_b', False, True)])
        with self.assertRaises(ValueError):
            read_scope('unrecognized input')



class ReviewRecordTest(unittest.TestCase):
    def test_repository_records_reference_in_scope_mnemonics_and_evidence(self):
        root = Path(__file__).resolve().parents[1]
        scope = read_scope((root / 'ISA_INVENTORY.md').read_text())
        reviews = read_reviews(root, {name for name, _, _ in scope})
        self.assertIn('loose contract reviewed', reviews['v_cvt_f32_i32'])
        self.assertIn('loose and exact contract reviewed', reviews['v_cvt_i32_i16'])
        self.assertNotIn('v_cvt_pk_u8_f32', reviews)

    def test_invalid_records_fail_instead_of_claiming_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tools').mkdir()
            (root / 'COVERAGE.md').write_text('## Reviewed family\n')
            (root / 'test.cpp').write_text('')
            record = dict(heading='Reviewed family', semantics='loose',
                          tests='test.cpp', mnemonics=['v_a'])
            path = root / 'tools/architecture_reviews.json'
            path.write_text(json.dumps([record]))
            self.assertIn('COVERAGE.md#reviewed-family', read_reviews(root, {'v_a'})['v_a'])
            for change in (dict(heading='Missing'), dict(semantics='complete'),
                           dict(tests='missing.cpp'), dict(mnemonics=['v_missing']),
                           dict(mnemonics=['v_a', 'v_a'])):
                with self.subTest(change=change):
                    path.write_text(json.dumps([dict(record, **change)]))
                    with self.assertRaises(ValueError):
                        read_reviews(root, {'v_a'})
            path.write_text(json.dumps([record, record]))
            with self.assertRaises(ValueError):
                read_reviews(root, {'v_a'})


if __name__ == '__main__':
    unittest.main()

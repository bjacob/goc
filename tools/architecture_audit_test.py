#!/usr/bin/env python3
"""Tests for review-queue extraction, not GPU semantic equivalence."""
import unittest
import sys

sys.dont_write_bytecode = True
from architecture_audit import direct_policies, execution_functions, read_scope


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


if __name__ == '__main__':
    unittest.main()

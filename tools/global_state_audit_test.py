#!/usr/bin/env python3
"""Check architecture resolution without asserting unsupported semantic coverage."""
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
from global_state_audit import generate, reporting_rows


class GlobalStateAuditTest(unittest.TestCase):
    def test_split_exclusive_and_unresolved_apis(self):
        inventory = '''| `v_split` | yes | yes | in | x | — | pending |
| `v_old` | yes | — | in | x | — | pending |
| `v_new` | — | yes | in | x | — | pending |
| `v_shared` | yes | yes | in | x | — | pending |'''
        header = '''int goc_v_split(uint32_t *trapsts);
int goc_v_split_rdna4(uint32_t *excp_flag_user);
int goc_v_split_wave64(uint32_t *trapsts);
int goc_v_split_rdna4_wave64(uint32_t *excp_flag_user);
int goc_v_old(uint32_t *excp_flag_user);
int goc_v_new(uint32_t *excp_flag_user);
int goc_v_shared(uint32_t *excp_flag_user);
int goc_v_unknown(uint32_t *trapsts);
int goc_v_no_reporting(uint32_t *d);
int goc_v_input(const uint32_t *trapsts);'''
        rows = {name: (arch, output, status) for name, arch, output, status
                in reporting_rows(header, inventory)}
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows['v_old'], ('RDNA3', 'excp_flag_user', 'name mismatch'))
        self.assertEqual(rows['v_split_wave64'], ('RDNA3', 'trapsts', 'name matches'))
        self.assertEqual(rows['v_split_rdna4_wave64'], ('RDNA4', 'excp_flag_user', 'name matches'))
        self.assertEqual(rows['v_new'], ('RDNA4', 'excp_flag_user', 'name matches'))
        self.assertEqual(rows['v_shared'][2], 'review required')
        self.assertEqual(rows['v_unknown'][2], 'review required')

    def test_public_resolved_signatures_name_the_architectural_register(self):
        root = Path(__file__).resolve().parents[1]
        rows = reporting_rows((root / 'include/goc/detail/goc_instructions.h').read_text(),
                              (root / 'ISA_INVENTORY.md').read_text())
        self.assertTrue(any(row[1] == 'RDNA3' for row in rows))
        self.assertTrue(any(row[1] == 'RDNA4' for row in rows))
        self.assertEqual([row for row in rows if row[3] == 'name mismatch'], [])
        # Keep generated runtime contract tests covering both register spellings.
        cmake = (root / 'tests/CMakeLists.txt').read_text()
        self.assertIn('(excp_flag_user|trapsts)', cmake)

    def test_generated_report_matches_current_signatures(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual((root / 'GLOBAL_STATE_AUDIT.md').read_text(), generate(root))


if __name__ == '__main__':
    unittest.main()

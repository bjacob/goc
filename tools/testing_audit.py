#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Report compiled instruction references and recorded test-oracle reviews.

Run after building tests: python3 tools/testing_audit.py BUILD_DIR --output FILE
Symbol references establish linkage, not execution or semantic coverage. Reviews
are explicit human evidence; this tool never infers oracle validity from names.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess

GENERIC = {
    'instruction_flags_test.cpp', 'exception_registers_test.cpp',
    'exec_mask_test.cpp', 'api_test.cpp', 'api_wave64_test.cpp',
}


def public_instructions(root):
    declarations = (root / 'include/goc/detail/goc_instructions.h').read_text()
    names = set(re.findall(r'^int (goc_\w+)\(', declarations, re.M))
    if not names:
        raise ValueError('No public instruction declarations found')
    return names


def active_objects(build, root):
    """Use exact compile outputs; never include abandoned objects left on disk."""
    import shlex
    objects = {}
    for entry in json.loads((build / 'compile_commands.json').read_text()):
        source = (Path(entry['directory']) / entry['file']).resolve()
        if source.parent != root / 'tests' or not source.name.endswith('_test.cpp'):
            continue
        args = entry.get('arguments') or shlex.split(entry['command'])
        output = entry.get('output')
        if output is None:
            output = args[args.index('-o') + 1]
        obj = (Path(entry['directory']) / output).resolve()
        if not obj.is_file():
            raise ValueError(f'Missing test object; build first: {obj}')
        objects[obj] = source.name
    if not objects:
        raise ValueError('No active test objects found')
    return objects


def compiled_references(objects):
    result = defaultdict(set)
    # Separate invocations avoid parsing platform-specific nm path delimiters.
    for obj, source in sorted(objects.items()):
        text = subprocess.check_output(['nm', '-u', str(obj)], text=True)
        for symbol in re.findall(r'\bU (goc_\w+)$', text, re.M):
            result[symbol].add(source)
    return result


def read_reviews(root):
    records = json.loads((root / 'tests/reference_reviews.json').read_text())
    reviewed = {}
    for record in records:
        file = record['file']
        if file in reviewed:
            raise ValueError(f'Duplicate reviewed test file: {file}')
        source = (root / 'tests' / file).read_text()
        for case in record['assertions']:
            suite, name = case.split('.', 1)
            pattern = rf'\bTEST(?:_P|_F)?\(\s*{re.escape(suite)}\s*,\s*{re.escape(name)}\s*\)'
            if not re.search(pattern, source):
                raise ValueError(f'Missing reviewed test assertion: {file}: {case}')
        for evidence in record['evidence']:
            if not (root / evidence).is_file():
                raise ValueError(f'Missing oracle evidence: {evidence}')
        if not record['assertions'] or not record['evidence'] or not record['scope']:
            raise ValueError(f'Empty oracle review: {file}')
        reviewed[file] = record
    return reviewed


def report(apis, refs, reviews):
    rows = []
    for api in sorted(apis):
        family = sorted(refs.get(api, set()) - GENERIC)
        rows.append({
            'api': api, 'family_tests': family,
            'generic_tests': sorted(refs.get(api, set()) & GENERIC),
            'reviewed_test_files': [f for f in family if f in reviews],
        })
    return {
        'limitations': 'Compiled references are not proof of execution or oracle equivalence. '
                        'Consult each review scope; file membership is not a per-mode proof.',
        'instructions': rows, 'reviews': list(reviews.values()),
        'missing_family_references': [r['api'] for r in rows if not r['family_tests']],
        'unreviewed_family_files': sorted({f for r in rows for f in r['family_tests']} - reviews.keys()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    objects = active_objects(args.build.resolve(), root)
    data = report(public_instructions(root), compiled_references(objects), read_reviews(root))
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(f"{len(data['instructions'])} instructions, {len(objects)} active test objects, "
          f"{len(data['missing_family_references'])} missing family references, "
          f"{len(data['unreviewed_family_files'])} family files awaiting oracle review")
    return bool(data['missing_family_references'])


if __name__ == '__main__':
    raise SystemExit(main())

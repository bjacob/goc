#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check capture_hwreg.py's raw gfx1201 register-access results.

Usage: python3 tests/check_hwreg_capture.py OUTPUT_DIR/capture.bin
Only MODE and EXCP_FLAG_USER user-mode field accesses are covered.
"""

import argparse
import hashlib
from pathlib import Path
import struct


def check(data):
    # RDNA4 section 3.4: MODE FP controls, FP16_OVFL, scalar prefetch,
    # performance disable; EXCP_FLAG_USER arithmetic flags, OOB, LOD_CLAMPED.
    registers = [("MODE", 0x098000FF), ("EXCP_FLAG_USER", 0xC000007F)]
    sources = [0, 0xFFFFFFFF, 0xA5A5A5A5, 0x5A5A5A5A, 0xFFFFFFFF]
    count = len(registers) * 2 * len(sources) * 32 * 32
    if len(data) != count * 12:
        raise ValueError(f"expected {count * 12} bytes, got {len(data)}")

    triples = iter(struct.iter_unpack("<III", data))
    for name, writable in registers:
        for seed in [0, 0xFFFFFFFF]:
            for pattern, source in enumerate(sources):
                for offset in range(32):
                    for width in range(1, 33):
                        before, after, field = next(triples)
                        initialized = seed & writable
                        field_mask = (1 << width) - 1
                        write_mask = (field_mask << offset) & writable
                        updated = ((initialized & ~write_mask) |
                                   ((source << offset) & write_mask))
                        extracted = (updated >> offset) & field_mask
                        expected = (initialized, updated, extracted)
                        actual = (before, after, field)
                        if actual != expected:
                            kind = "literal" if pattern == 4 else "SGPR"
                            raise ValueError(
                                f"{name} seed={seed:#010x} {kind} "
                                f"source={source:#010x} offset={offset} "
                                f"width={width}: expected "
                                f"{tuple(hex(x) for x in expected)}, got "
                                f"{tuple(hex(x) for x in actual)}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    data = args.capture.read_bytes()
    try:
        count = check(data)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(f"Verified {count:,} register-access triples")
    print(f"SHA256 {hashlib.sha256(data).hexdigest()}")

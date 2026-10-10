#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check exhaustive capture_half_unary.py rounding/FREXP_MANT/FRACT captures.

Usage: check_half_unary_capture.py MNEMONIC CAPTURE_BIN
Requires NumPy. Mathematical operations use exactly represented binary64 inputs;
raw-bit overrides retain NaN payloads and the observed instruction exceptions.
This is a capture-analysis tool, not a production emulation path.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np


def reference(op, saturate, policy, modifier):
    bits = np.arange(65536, dtype=np.uint16)
    if modifier & 2:
        bits &= 0x7FFF
    if modifier & 1:
        bits ^= 0x8000
    magnitude = bits & 0x7FFF
    subnormal = (magnitude != 0) & (magnitude < 0x400)
    snan = (magnitude > 0x7C00) & ((bits & 0x200) == 0)
    exceptions = snan.astype(np.uint32)
    if op in ("frexp_mant", "fract"):
        exceptions |= (magnitude == 0x7C00).astype(np.uint32)
    if policy & 1:
        bits = np.where(subnormal, bits & 0x8000, bits).astype(np.uint16)
    else:
        exceptions |= subnormal.astype(np.uint32) * 2

    # Every finite half input and x - floor(x) are exact in binary64.
    # FRACT rounds to half and caps below one before output scaling.
    value = bits.view(np.float16).astype(np.float64)
    if op == "fract":
        result = value - np.floor(value)
        rounded = result.astype(np.float16).astype(np.float64)
        if not modifier & 12:
            exceptions |= ((rounded != result) & np.isfinite(value)).astype(np.uint32) * 32
        result = np.minimum(rounded, 1 - 2**-11)
        tiny = (result != 0) & (result < 2**-14)
        if policy & 2 and not modifier & 12:
            result = np.where(tiny, 0, result)
            exceptions |= tiny.astype(np.uint32) * 48
    elif op == "frexp_mant":
        result = np.frexp(value)[0]
    else:
        result = {"trunc": np.trunc, "ceil": np.ceil, "rndne": np.rint,
                  "floor": np.floor}[op](value)
    if modifier & 12:
        # FP16 OMOD's input flush canonicalizes tiny values, including -0, to +0.
        result = np.where(np.abs(result) < 2**-14, 0, result)
    result *= [1, 2, 4, 0.5][(modifier >> 2) & 3]
    if saturate:
        result = np.where(np.isfinite(result), np.clip(result, -65504, 65504), result)
    if op == "fract" and modifier & 12:
        result = np.where(np.abs(result) < 2**-14, 0, result)
    output = result.astype(np.float16).view(np.uint16)
    if op == "fract":
        output = np.where(magnitude == 0x7C00, 0xFE00, output).astype(np.uint16)
    nan = magnitude > 0x7C00
    output = np.where(nan, bits | 0x200, output).astype(np.uint16)
    if modifier & 16:
        output = np.where(nan | ((output & 0x8000) != 0), 0,
                          np.minimum(output, 0x3C00)).astype(np.uint16)
        exceptions.fill(0)
    return output, exceptions


def check(op, path):
    expected_bytes = 2 * 4 * 32 * 65536 * 8
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"expected {expected_bytes} bytes, got {path.stat().st_size}")
    capture = np.memmap(path, dtype="<u4", mode="r").reshape(2, 4, 32, 65536, 2)
    with np.errstate(all="ignore"):
        for saturate in range(2):
            for policy in range(4):
                for modifier in range(32):
                    expected, exceptions = reference(op, saturate, policy, modifier)
                    actual = capture[saturate, policy, modifier]
                    mismatches = np.flatnonzero((actual[:, 0] != expected) |
                                               (actual[:, 1] != exceptions))
                    if len(mismatches):
                        i = int(mismatches[0])
                        raise ValueError(
                            f"{op} saturate={saturate} policy={policy} modifier={modifier} "
                            f"input={i:#06x}: expected ({int(expected[i]):#06x}, "
                            f"{int(exceptions[i]):#x}), got ({int(actual[i, 0]):#06x}, "
                            f"{int(actual[i, 1]):#x}); {len(mismatches)} mismatches in group")
    return expected_bytes // 8


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mnemonic", choices=["trunc", "ceil", "rndne", "floor", "frexp_mant", "fract"])
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    try:
        count = check(args.mnemonic, args.capture)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(f"{args.mnemonic}: verified {count:,} raw result/exception pairs")
    print(f"SHA256 {hashlib.sha256(args.capture.read_bytes()).hexdigest()}")

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independently check exhaustive capture_half_unary.py RCP/RSQ/SQRT captures.

Usage: check_half_sfu_capture.py MNEMONIC CAPTURE_BIN
Requires NumPy. Binary64 mathematical operations and FP16 narrowing check the
results independently of GoC/rocjitsu's empirical FP32 approximation models.
Exception classification follows the exhaustively captured gfx1201 rules.
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
    nan = magnitude > 0x7C00
    if policy & 1:
        bits = np.where(subnormal, bits & 0x8000, bits).astype(np.uint16)
    value = bits.view(np.float16).astype(np.float64)
    finite = np.isfinite(value)
    negative = (bits & 0x8000) != 0
    invalid = (nan & ((magnitude & 0x200) == 0)) | (
        (op != "rcp") & negative & (value != 0) & ~nan)
    exceptions = invalid.astype(np.uint32)
    if not policy & 1:
        exceptions |= (subnormal & ~invalid).astype(np.uint32) * 2
    if op != "sqrt":
        exceptions |= (value == 0).astype(np.uint32) * 4

    if op == "rcp":
        result = 1 / value
    elif op == "rsq":
        result = 1 / np.sqrt(value)
    else:
        result = np.sqrt(value)
    rounded = result.astype(np.float16).astype(np.float64)
    mantissa, exponent = np.frexp(np.abs(value))
    exact = mantissa == 0.5
    if op != "rcp":
        # Hardware's approximation unit reports INEXACT even for sqrt(9).
        # Only powers of four avoid it for positive finite SQRT/RSQ inputs.
        exact &= (exponent & 1) != 0
    regular = finite & (value != 0) & ~invalid
    overflow = regular & np.isinf(rounded)
    exceptions |= overflow.astype(np.uint32) * 8
    if not modifier & 12:
        inexact = regular & (~exact | overflow)
        tiny = regular & (np.abs(rounded) < 2**-14)
        if policy & 2:
            inexact |= tiny
        exceptions |= inexact.astype(np.uint32) * 32
        exceptions |= (tiny & inexact).astype(np.uint32) * 16

    # Saturation includes infinity generated from zero, but not an input infinity.
    # Architectural rounding precedes OMOD. Guest output flushing preserves sign.
    if saturate:
        rounded = np.where(finite, np.clip(rounded, -65504, 65504), rounded)
    if policy & 2:
        rounded = np.where(np.abs(rounded) < 2**-14, np.copysign(0., rounded), rounded)
    if modifier & 12:
        rounded = np.where(np.abs(rounded) < 2**-14, 0, rounded)
    scaled = rounded * [1, 2, 4, 0.5][(modifier >> 2) & 3]
    scaling_overflow = np.isfinite(rounded) & np.isinf(scaled.astype(np.float16))
    if op == "rsq":
        # Unlike RCP, saturated RSQ zero/flushed-zero retains only DIV0 when
        # OMOD overflows the finite saturation value. Both were captured.
        scaling_overflow &= value != 0
    exceptions |= scaling_overflow.astype(np.uint32) * 8
    if saturate:
        scaled = np.where(np.isfinite(scaled), np.clip(scaled, -65504, 65504), scaled)
    if modifier & 12:
        # Input-to-OMOD tiny values become +0 above, but newly tiny results
        # from div:2 preserve sign (observable for negative reciprocals).
        scaled = np.where(np.abs(scaled) < 2**-14, np.copysign(0., scaled), scaled)
    output = scaled.astype(np.float16).view(np.uint16)
    output = np.where(invalid & ~nan, 0xFE00, output).astype(np.uint16)
    output = np.where(nan, bits | 0x200, output).astype(np.uint16)
    if modifier & 16:
        output = np.where(((output & 0x8000) != 0) | ((output & 0x7FFF) > 0x7C00),
                          0, np.minimum(output, 0x3C00)).astype(np.uint16)
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
    parser.add_argument("mnemonic", choices=["rcp", "rsq", "sqrt"])
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    try:
        count = check(args.mnemonic, args.capture)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(f"{args.mnemonic}: verified {count:,} raw result/exception pairs")
    print(f"SHA256 {hashlib.sha256(args.capture.read_bytes()).hexdigest()}")

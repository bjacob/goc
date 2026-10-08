// SPDX-License-Identifier: MIT

#pragma once

namespace goc {

// Placement of the scalar literal in a fused multiply-add.
enum class FmaOperands { Registers, MultiplyLiteral, AddLiteral };

} // namespace goc

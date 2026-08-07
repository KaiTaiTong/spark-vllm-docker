#!/usr/bin/env python3
"""Preserve explicitly selected CUDA 13 SM12x subarchitecture targets."""

import os
import sys
from pathlib import Path


if os.environ.get("VLLM_PRESERVE_SM12X_TARGET", "") not in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}:
    print("SM12x target preservation not requested; skipping")
    raise SystemExit(0)

source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
target = source_root / "CMakeLists.txt"
cuda13_default = (
    'set(CUDA_SUPPORTED_ARCHS '
    '"7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")'
)
cuda13_sm121 = (
    'set(CUDA_SUPPORTED_ARCHS '
    '"7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0;12.1")'
)

text = target.read_text()
if cuda13_sm121 in text:
    print("CUDA 13 SM12x subarchitecture allow-list already present; skipping")
elif text.count(cuda13_default) == 1:
    target.write_text(text.replace(cuda13_default, cuda13_sm121, 1))
    print("Enabled selected SM12x target preservation for CUDA 13 vLLM build")
else:
    raise SystemExit(
        "Unable to preserve the selected SM12x target: expected CUDA 13 "
        "CUDA_SUPPORTED_ARCHS entry was not found exactly once"
    )

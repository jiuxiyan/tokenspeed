#!/usr/bin/env bash
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Source this file from bash after explicitly setting TASK1_CHIP.

if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "Source env.sh from bash." >&2
    return 1 2>/dev/null || exit 1
fi

case "${TASK1_CHIP:-}" in
    ''|*[!0-9]*)
        echo "Set TASK1_CHIP to one numeric logical chip index before sourcing env.sh." >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

TASK1_REPO_ROOT="${TASK1_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
TASK1_VENV="${TASK1_VENV:-${HOME}/task1/.venv}"
TASK1_CANN_ROOT="${TASK1_CANN_ROOT:-/usr/local/Ascend/cann}"

if [[ ! -r "${TASK1_CANN_ROOT}/set_env.sh" || ! -r "${TASK1_VENV}/bin/activate" ]]; then
    echo "TASK1_CANN_ROOT/set_env.sh and TASK1_VENV/bin/activate must exist." >&2
    return 1 2>/dev/null || exit 1
fi

TASK1_RESTORE_NOUNSET=0
case "$-" in
    *u*) TASK1_RESTORE_NOUNSET=1; set +u ;;
esac
source "${TASK1_CANN_ROOT}/set_env.sh"
source "${TASK1_VENV}/bin/activate"
if [[ "${TASK1_RESTORE_NOUNSET}" == 1 ]]; then
    set -u
fi
export ASCEND_RT_VISIBLE_DEVICES="${TASK1_CHIP}"
export PYTHONPATH="${TASK1_REPO_ROOT}:${TASK1_REPO_ROOT}/python:${TASK1_REPO_ROOT}/tokenspeed-kernel/python:${TASK1_REPO_ROOT}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_DEACTIVATE_ASYNC_LOAD=1

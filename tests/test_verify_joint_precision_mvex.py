from __future__ import annotations

import hashlib
import json

import pytest

from scripts.analyze.verify_a2_reproduction import VerificationError
from scripts.analyze.verify_joint_precision_mvex import (
    canonical_sample_contract_sha,
    command_value,
    fallacy_scan,
    request_accounting,
)


def test_canonical_sample_contract_hash_excludes_command_and_embedded_hash() -> None:
    base = {"sample_id": "full__random__r30__s7", "seed": 7}
    payload = json.dumps(base, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    contract = {**base, "contract_sha256": "ignored", "command": ["ignored"]}

    assert canonical_sample_contract_sha(contract) == hashlib.sha256(payload).hexdigest()


def test_request_accounting_fails_closed_on_reduced_denominator() -> None:
    result = {
        "completed": 2,
        "failed": 0,
        "ttfts": [0.1, 0.2],
        "itls": [[], []],
        "input_lens": [1, 1],
        "output_lens": [1, 1],
        "start_times": [0.0, 1.0],
        "errors": ["", ""],
    }

    with pytest.raises(VerificationError, match="frozen denominator"):
        request_accounting(result, 3)


def test_command_value_rejects_duplicate_flags() -> None:
    with pytest.raises(VerificationError, match="exactly one"):
        command_value(["tool", "--seed", "7", "--seed", "42"], "--seed")


def test_fallacy_scan_covers_all_eleven_categories() -> None:
    scan = fallacy_scan()

    assert len(scan) == 11
    assert len({item["fallacy"] for item in scan}) == 11

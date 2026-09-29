"""Private versioned-JSON helpers shared by Task 6 risk contracts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime
from numbers import Integral, Real
from typing import Any

_SCHEMA_VERSION = 1


def aware_datetime(name: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def parse_datetime(name: str, value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO-8601 datetime") from exc
    return aware_datetime(name, parsed)


def required_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty without surrounding whitespace")
    return value


def optional_text(name: str, value: object) -> str | None:
    if value is None:
        return None
    return required_text(name, value)


def finite_float(name: str, value: object, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return result


def positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_mapping(text: str, contract_name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{contract_name} JSON is invalid") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{contract_name} JSON must contain an object")
    return value


def require_payload(
    data: object, *, contract_name: str, fields: frozenset[str]
) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{contract_name} payload must be a mapping")
    expected = fields | {"contract", "schema_version"}
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(repr(field) for field in actual - expected)
        raise ValueError(
            f"invalid {contract_name} payload: missing={missing}, unknown={unknown}"
        )
    if data["contract"] != contract_name:
        raise ValueError(f"expected contract={contract_name!r}")
    version = data["schema_version"]
    if type(version) is not int or version != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported {contract_name} schema_version {version!r}; "
            f"expected integer {_SCHEMA_VERSION}"
        )
    return data


def payload(contract_name: str, **fields: Any) -> dict[str, Any]:
    return {"contract": contract_name, "schema_version": _SCHEMA_VERSION, **fields}

"""Audited process-local MuJoCo arena capacity override, without editing assets.

Only <size memory/njmax/nstack> changes. In particular nconmax is preserved:
MuJoCo 3.3.2 still uses it as a contact limit, not merely a memory estimate.
Capacity changes require real scene validation and warning checks; this module
does not establish physics equivalence or a total-process memory bound.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
from unittest.mock import patch
from xml.parsers import expat
from xml.sax.saxutils import quoteattr


DEFAULT_MEMORY_BYTES = 256 * 1024**2
CAPACITY_ATTRIBUTES = frozenset({"memory", "njmax", "nstack"})


def _budget(memory_bytes):
    if type(memory_bytes) is not int or memory_bytes <= 0 or memory_bytes % 64:
        raise ValueError("Arena budget must be a positive integer number of bytes aligned to 64")
    return memory_bytes


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _size_tag(payload):
    parser = expat.ParserCreate("UTF-8")
    stack, found = [], []

    def start(name, attributes):
        if not stack and name != "mujoco":
            raise ValueError("Expected a mujoco XML root")
        if stack and stack[-1] == "size":
            raise ValueError("The size element must have no child elements")
        if name == "size":
            if stack != ["mujoco"]:
                raise ValueError("Only one direct-child size element is supported")
            found.append((parser.CurrentByteIndex, dict(attributes)))
        stack.append(name)

    def doctype(*unused):
        raise ValueError("DOCTYPE declarations are not supported for the arena override")

    parser.StartElementHandler = start
    parser.EndElementHandler = lambda name: stack.pop()
    parser.StartDoctypeDeclHandler = doctype
    try:
        parser.Parse(payload, True)
    except expat.ExpatError as error:
        raise ValueError("Malformed XML or duplicate attribute in arena input") from error
    if len(found) != 1:
        raise ValueError("Expected exactly one size element")
    first, attributes = found[0]
    quote = None
    for last in range(first, len(payload)):
        char = payload[last]
        if quote is not None:
            if char == quote:
                quote = None
        elif char in (34, 39):
            quote = char
        elif char == 62:
            return first, last + 1, attributes
    raise ValueError("Unterminated size opening tag")


def rewrite_xml(xml, *, memory_bytes=DEFAULT_MEMORY_BYTES):
    """Return (XML, audit), preserving every byte outside the size opening tag."""
    _budget(memory_bytes)
    if not isinstance(xml, str):
        raise TypeError("MjSim.from_xml_string input must be a Unicode XML string")
    payload = xml.encode("utf-8")
    first, last, old = _size_tag(payload)
    new = {name: value for name, value in old.items() if name not in CAPACITY_ATTRIBUTES}
    new["memory"] = str(memory_bytes)
    self_closing = payload[first:last].rstrip().endswith(b"/>")
    replacement = ("<size" + "".join(f" {name}={quoteattr(value)}" for name, value in new.items())
                   + ("/>" if self_closing else ">")).encode("utf-8")
    updated = payload[:first] + replacement + payload[last:]
    _, _, checked = _size_tag(updated)
    if checked != new:
        raise ValueError("Arena rewrite did not preserve size attribute semantics")
    audit = {"input_sha256": _digest(payload), "output_sha256": _digest(updated),
             "input_bytes": len(payload), "output_bytes": len(updated),
             "old_size": old, "new_size": new, "memory_budget_bytes": memory_bytes,
             "changed_attributes": sorted(name for name in set(old) | set(new) if old.get(name) != new.get(name)),
             "size_opening_tag_input_byte_range": [first, last],
             "size_opening_tag_output_byte_range": [first, first + len(replacement)],
             "outside_size_opening_tag_sha256": _digest(payload[:first] + payload[last:]),
             "outside_size_opening_tag_bytes_unchanged": True,
             "nconmax_preserved": old.get("nconmax") == new.get("nconmax"),
             "scope": "Arena allocation capacity only; physics equivalence requires scene tests and zero warnings"}
    return updated.decode("utf-8"), audit


def check_simulator_capacity(simulator, *, memory_bytes=DEFAULT_MEMORY_BYTES):
    """Read current native allocation/warnings; call again during actual rollout.

    Accept the audited robosuite MjSim wrapper or an object with native MuJoCo
    model/data. No model, renderer, Torch, or environment is imported here.
    """
    _budget(memory_bytes)
    model = getattr(simulator.model, "_model", simulator.model)
    data = getattr(simulator.data, "_data", simulator.data)
    record = {"model_arena_bytes": int(model.narena), "data_arena_bytes": int(data.narena),
              "maximum_used_arena_bytes": int(data.maxuse_arena),
              "warning_counts": [int(item.number) for item in data.warning],
              "warning_lastinfo": [int(item.lastinfo) for item in data.warning]}
    if not 0 < record["model_arena_bytes"] == record["data_arena_bytes"] <= memory_bytes:
        raise RuntimeError("Compiled MuJoCo arena does not respect the explicit byte budget")
    if not 0 <= record["maximum_used_arena_bytes"] <= record["data_arena_bytes"]:
        raise RuntimeError("MuJoCo arena high-water mark exceeds its allocation")
    if any(record["warning_counts"]):
        raise RuntimeError(f"MuJoCo warning counters are nonzero: {record['warning_counts']}")
    return record


@contextmanager
def arena_budget(mjsim_class, *, memory_bytes=DEFAULT_MEMORY_BYTES):
    """Patch the audited @classmethod MjSim.from_xml_string in this process only.

    All compilations through this class, including from_xml_file delegation, are
    recorded. The original descriptor is restored even if compilation fails.
    Callers retain audit and persist it outside this context for crash evidence.
    """
    _budget(memory_bytes)
    descriptor = inspect.getattr_static(mjsim_class, "from_xml_string")
    if not isinstance(descriptor, classmethod):
        raise TypeError("Expected the audited robosuite classmethod from_xml_string")
    import mujoco

    audit = {"memory_budget_bytes": memory_bytes, "descriptor": "classmethod",
             "nconmax_policy": "preserve original contact limit",
             "native_warning_policy": "record and reject every native warning; restore prior callback on exit",
             "native_warnings": [], "compilations": []}
    previous_warning_callback = mujoco.get_mju_user_warning()

    def reject_warning(message):
        audit["native_warnings"].append({"message": str(message),
                                         "latest_compile_index": len(audit["compilations"]) - 1})
        raise RuntimeError(f"MuJoCo native warning rejected by arena experiment: {message}")

    @wraps(descriptor.__func__)
    def compile_xml(cls, xml):
        updated, record = rewrite_xml(xml, memory_bytes=memory_bytes)
        record.update(compile_index=len(audit["compilations"]), status="compiling")
        audit["compilations"].append(record)
        try:
            result = descriptor.__func__(cls, updated)
            record["compiled_memory"] = check_simulator_capacity(result, memory_bytes=memory_bytes)
            record["status"] = "compiled"
            return result
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__)
            raise

    mujoco.set_mju_user_warning(reject_warning)
    try:
        with patch.object(mjsim_class, "from_xml_string", classmethod(compile_xml)):
            yield audit
    finally:
        mujoco.set_mju_user_warning(previous_warning_callback)

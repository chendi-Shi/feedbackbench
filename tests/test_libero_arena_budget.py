"""XML contracts plus a tiny native MuJoCo dynamics check; no LIBERO or Torch."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/libero_arena_budget.py"
SPEC = importlib.util.spec_from_file_location("libero_arena_budget_test", PATH)
budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)

XML = """<mujoco model="tiny-arena-check">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="Euler"/>
  <size njmax="64" nstack="10000" nconmax="5000" nuserdata="2"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.25"><freejoint/><geom type="sphere" size="0.08" mass="1"/></body>
  </worldbody>
</mujoco>"""


def test_import_has_no_model_or_graphics_dependencies():
    result = subprocess.run([sys.executable, "-c",
                             "import runpy,sys;runpy.run_path(sys.argv[1]);"
                             "assert not ({'torch','mujoco','robosuite','libero'} & sys.modules.keys())", str(PATH)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_only_capacity_attributes_change_and_contact_limit_is_preserved():
    updated, audit = budget.rewrite_xml(XML)
    old_root, new_root = ET.fromstring(XML), ET.fromstring(updated)
    assert new_root.find("size").attrib == {"nconmax": "5000", "nuserdata": "2", "memory": "268435456"}
    assert audit["changed_attributes"] == ["memory", "njmax", "nstack"]
    assert audit["old_size"]["nconmax"] == audit["new_size"]["nconmax"] == "5000"
    assert audit["input_sha256"] == hashlib.sha256(XML.encode()).hexdigest()
    first, last = audit["size_opening_tag_input_byte_range"]
    new_first, new_last = audit["size_opening_tag_output_byte_range"]
    assert XML.encode()[:first] == updated.encode()[:new_first]
    assert XML.encode()[last:] == updated.encode()[new_last:]
    old_root.remove(old_root.find("size"))
    new_root.remove(new_root.find("size"))
    assert ET.tostring(old_root) == ET.tostring(new_root)


def test_comments_unicode_entities_and_non_self_closing_size_do_not_confuse_patch():
    xml = '<mujoco><!-- 中文 <size memory="1"/> --><size nconmax="&#53;000" nuserdata="2"></size><worldbody/></mujoco>'
    updated, audit = budget.rewrite_xml(xml)
    assert '<!-- 中文 <size memory="1"/> -->' in updated
    assert '<size nconmax="5000" nuserdata="2" memory="268435456"></size>' in updated
    assert audit["outside_size_opening_tag_bytes_unchanged"] is True


def test_existing_memory_is_replaced_without_touching_other_size_attributes():
    xml = '<mujoco><size memory="4M" nconmax="7" nuser_geom="3"/><worldbody/></mujoco>'
    updated, audit = budget.rewrite_xml(xml)
    assert ET.fromstring(updated).find("size").attrib == {"memory": "268435456", "nconmax": "7", "nuser_geom": "3"}
    assert audit["changed_attributes"] == ["memory"]


@pytest.mark.parametrize("xml", [
    '<mujoco><size memory="1" memory="2"/></mujoco>',
    '<mujoco><size/><size/></mujoco>',
    '<mujoco><worldbody/></mujoco>',
    '<mujoco><worldbody><size/></worldbody></mujoco>',
    '<mujoco><size><body/></size></mujoco>',
    '<mujoco><size></mujoco>',
    '<other><size/></other>',
    '<!DOCTYPE mujoco [<!ENTITY capacity "1">]><mujoco><size memory="&capacity;"/></mujoco>',
])
def test_ambiguous_or_malformed_xml_is_rejected(xml):
    with pytest.raises(ValueError):
        budget.rewrite_xml(xml)


@pytest.mark.parametrize("value", [True, 0, -64, 1, 64.0, "268435456"])
def test_invalid_budget_is_rejected(value):
    with pytest.raises(ValueError):
        budget.rewrite_xml(XML, memory_bytes=value)


def fake_sim(memory_bytes=budget.DEFAULT_MEMORY_BYTES, warning=0):
    return SimpleNamespace(model=SimpleNamespace(narena=memory_bytes),
                           data=SimpleNamespace(narena=memory_bytes, maxuse_arena=1024,
                                                warning=[SimpleNamespace(number=warning, lastinfo=0)]))


def test_classmethod_subclass_binding_multiple_compilations_and_restoration():
    calls = []

    class Sim:
        @classmethod
        def from_xml_string(cls, xml):
            calls.append((cls, xml))
            return fake_sim()

        @classmethod
        def from_xml_file(cls, xml_file):
            return cls.from_xml_string(xml_file)

    class Child(Sim):
        pass

    original = inspect.getattr_static(Sim, "from_xml_string")
    with budget.arena_budget(Sim) as audit:
        Sim.from_xml_string(XML)
        Child.from_xml_file(XML)
        assert [row[0] for row in calls] == [Sim, Child]
        assert len(audit["compilations"]) == 2
        assert all(row["status"] == "compiled" for row in audit["compilations"])
        assert all(row["compiled_memory"]["warning_counts"] == [0] for row in audit["compilations"])
        assert all('memory="268435456"' in row[1] for row in calls)
    assert inspect.getattr_static(Sim, "from_xml_string") is original


def test_compile_exception_is_audited_and_original_descriptor_restored():
    class Sim:
        @classmethod
        def from_xml_string(cls, xml):
            raise RuntimeError("controlled compile error")

    original = inspect.getattr_static(Sim, "from_xml_string")
    with pytest.raises(RuntimeError, match="controlled"):
        with budget.arena_budget(Sim) as audit:
            Sim.from_xml_string(XML)
    assert inspect.getattr_static(Sim, "from_xml_string") is original
    assert audit["compilations"][0]["status"] == "failed"
    assert audit["compilations"][0]["error_type"] == "RuntimeError"


def test_descriptor_drift_is_rejected():
    class Sim:
        @staticmethod
        def from_xml_string(xml):
            return fake_sim()

    with pytest.raises(TypeError, match="classmethod"):
        with budget.arena_budget(Sim):
            pytest.fail("Unsupported descriptor accepted")


@pytest.mark.parametrize("during_compile", [False, True])
def test_native_warning_is_recorded_rejected_and_previous_callback_restored(during_compile):
    import mujoco

    def native_warning():
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><size memory="1048576"/><worldbody><body><freejoint/>'
            '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>')
        data = mujoco.MjData(model)
        data.qpos[0] = float("nan")
        mujoco.mj_step(model, data)

    class Sim:
        @classmethod
        def from_xml_string(cls, xml):
            if during_compile:
                native_warning()
            return fake_sim()

    original_callback = mujoco.get_mju_user_warning()
    previous_messages = []

    def previous_callback(message):
        previous_messages.append(message)

    original_descriptor = inspect.getattr_static(Sim, "from_xml_string")
    mujoco.set_mju_user_warning(previous_callback)
    try:
        with pytest.raises(RuntimeError, match="native warning rejected"):
            with budget.arena_budget(Sim) as audit:
                Sim.from_xml_string(XML)
                native_warning()
        assert mujoco.get_mju_user_warning() is previous_callback
        assert inspect.getattr_static(Sim, "from_xml_string") is original_descriptor
        assert len(audit["native_warnings"]) == 1
        assert audit["native_warnings"][0]["latest_compile_index"] == 0
        assert previous_messages == []
        assert audit["compilations"][0]["status"] == ("failed" if during_compile else "compiled")
        native_warning()
        assert len(previous_messages) == 1 and "QPOS" in previous_messages[0]
    finally:
        mujoco.set_mju_user_warning(original_callback)


def test_warning_callback_restores_after_normal_context_exit():
    import mujoco

    class Sim:
        @classmethod
        def from_xml_string(cls, xml):
            return fake_sim()

    original_callback = mujoco.get_mju_user_warning()
    with budget.arena_budget(Sim) as audit:
        Sim.from_xml_string(XML)
    assert mujoco.get_mju_user_warning() is original_callback
    assert audit["native_warnings"] == []


@pytest.mark.parametrize("kind", ["over_budget", "warning", "high_water"])
def test_capacity_and_warning_fail_closed(kind):
    sim = fake_sim()
    if kind == "over_budget":
        sim.model.narena = sim.data.narena = budget.DEFAULT_MEMORY_BYTES + 64
    elif kind == "warning":
        sim.data.warning[0].number = 1
    else:
        sim.data.maxuse_arena = budget.DEFAULT_MEMORY_BYTES + 1
    with pytest.raises(RuntimeError):
        budget.check_simulator_capacity(sim)


def test_real_mujoco_contact_dynamics_match_with_explicit_arena_and_no_warnings():
    import mujoco
    import numpy as np

    assert mujoco.__version__ == "3.3.2"
    cap = 8 * 1024**2
    updated, _ = budget.rewrite_xml(XML, memory_bytes=cap)
    original_model = mujoco.MjModel.from_xml_string(XML)
    revised_model = mujoco.MjModel.from_xml_string(updated)
    original_data, revised_data = mujoco.MjData(original_model), mujoco.MjData(revised_model)
    assert original_model.nconmax == revised_model.nconmax == 5000
    assert revised_model.narena == revised_data.narena == cap
    simulation = SimpleNamespace(model=revised_model, data=revised_data)
    saw_contact = False
    for _ in range(200):
        mujoco.mj_step(original_model, original_data)
        mujoco.mj_step(revised_model, revised_data)
        assert np.array_equal(original_data.qpos, revised_data.qpos)
        assert np.array_equal(original_data.qvel, revised_data.qvel)
        assert np.array_equal(original_data.qacc, revised_data.qacc)
        assert original_data.ncon == revised_data.ncon
        assert original_data.nefc == revised_data.nefc
        assert original_data.time == revised_data.time
        assert not any(item.number for item in original_data.warning)
        budget.check_simulator_capacity(simulation, memory_bytes=cap)
        saw_contact |= bool(original_data.ncon)
    assert saw_contact
    assert 0 < revised_data.maxuse_arena < cap
    assert "torch" not in sys.modules

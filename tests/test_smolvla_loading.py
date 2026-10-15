"""Explicit CPU integration check: pytest tests/test_smolvla_loading.py -q.

Loads the complete public 605M-parameter checkpoint once (~2.6 GiB resident),
so run sequentially with other SmolVLA profiles on the 16 GB target computer.
"""

from embodied_research.policies import PolicyAdapter


def test_full_smolvla_checkpoint_and_nonpersistent_buffers():
    import torch

    torch.set_num_threads(2)
    policy = PolicyAdapter("smolvla")
    assert policy.metadata["parameters"] == 604934176
    assert policy.metadata["strict_checkpoint_load"]
    assert policy.metadata["skipped_discarded_random_initialization"]
    audit = policy.metadata["loading_audit"]
    assert audit["checkpoint_state_tensors"] == 788
    assert audit["exact_state_keys_and_shapes"]
    assert len(audit["nonpersistent_buffers"]) == 4
    assert all(record["matches_closed_form"] for record in audit["nonpersistent_buffers"].values())

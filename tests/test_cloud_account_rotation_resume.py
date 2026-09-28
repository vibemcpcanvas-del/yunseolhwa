"""Verification tests for Gate 3: Verified Step Resumption on Colab Account Rotation."""

import json
import os
import shutil
import tarfile
import tempfile
import pytest

from cloud.cloud_manager import CloudJaxManager, package_jax_codebase
from cloud.colab_tunnel_bridge import JaxTunnelTransferManager


@pytest.fixture
def test_workspace():
    d = tempfile.mkdtemp(prefix="test_rotation_resume_")
    yield d
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)


def test_unverified_steps_never_selected(test_workspace):
    """Test 1: Unverified steps or steps failed in transit are never marked for resumption."""
    manager = JaxTunnelTransferManager(
        data_tar_path=os.path.join(test_workspace, "data.tar"),
        script_path=os.path.join(test_workspace, "job.py"),
        output_checkpoints_tar=os.path.join(test_workspace, "out.tar"),
        extract_target_dir=os.path.join(test_workspace, "checkpoints"),
    )

    # Initial state
    assert manager.get_last_verified_step() == 0
    assert not manager.is_step_verified(50)

    # Add verified step 50
    manager.verified_steps.add(50)
    assert manager.get_last_verified_step() == 50
    assert manager.is_step_verified(50)

    # Unverified step 100
    assert not manager.is_step_verified(100)
    assert manager.get_last_verified_step() == 50


def test_package_jax_codebase_with_resume_checkpoint(test_workspace):
    """Test 2: package_jax_codebase packages resume_step_dir into resume_checkpoint/."""
    # Create mock resume checkpoint dir
    ckpt_dir = os.path.join(test_workspace, "step_50")
    resume_dir = os.path.join(ckpt_dir, "_resume_state")
    os.makedirs(resume_dir, exist_ok=True)

    manifest_data = {"schema_version": 1, "update": 50, "mode": 1}
    with open(os.path.join(resume_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    out_tar = os.path.join(test_workspace, "upload_jax.tar")
    package_jax_codebase(out_tar, resume_step_dir=ckpt_dir)

    # Inspect tarball contents
    with tarfile.open(out_tar, "r") as tar:
        names = tar.getnames()
        # Must contain resume_checkpoint/_resume_state/manifest.json
        assert any("resume_checkpoint" in name for name in names)
        assert any("manifest.json" in name for name in names)

    # Extract to target directory and verify exact structure expected by colab_jax_train_job.py
    extracted_workspace = os.path.join(test_workspace, "extracted_workspace")
    os.makedirs(extracted_workspace, exist_ok=True)
    with tarfile.open(out_tar, "r") as tar:
        tar.extractall(path=extracted_workspace)

    extracted_manifest = os.path.join(
        extracted_workspace, "resume_checkpoint", "_resume_state", "manifest.json"
    )
    assert os.path.exists(extracted_manifest)
    with open(extracted_manifest, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["update"] == 50


def test_package_jax_codebase_fresh_start_no_resume(test_workspace):
    """Test 3: package_jax_codebase without resume_step_dir omits resume_checkpoint."""
    out_tar = os.path.join(test_workspace, "upload_fresh.tar")
    package_jax_codebase(out_tar, resume_step_dir=None)

    with tarfile.open(out_tar, "r") as tar:
        names = tar.getnames()
        assert not any("resume_checkpoint" in name for name in names)


def test_cloud_jax_manager_get_verified_resume_dir(test_workspace, monkeypatch):
    """Test 4: CloudJaxManager locates valid checkpoint on disk only when verified."""
    mgr = CloudJaxManager()
    assert mgr.get_verified_resume_dir(mode=1) is None

    # Mock checkpoints dir
    mock_ckpt_root = os.path.join(test_workspace, "checkpoints")
    step_dir = os.path.join(mock_ckpt_root, "mode_1", "step_100")
    resume_dir = os.path.join(step_dir, "_resume_state")
    os.makedirs(resume_dir, exist_ok=True)
    with open(os.path.join(resume_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"schema_version": 1, "update": 100}, f)

    # Patch PROJECT_ROOT in cloud_manager
    import cloud.cloud_manager as cm
    monkeypatch.setattr(cm, "PROJECT_ROOT", test_workspace)

    mgr.last_verified_step = 100
    found_dir = mgr.get_verified_resume_dir(mode=1)
    assert found_dir == step_dir

    # Check unverified / non-existent step
    mgr.last_verified_step = 200
    assert mgr.get_verified_resume_dir(mode=1) is None

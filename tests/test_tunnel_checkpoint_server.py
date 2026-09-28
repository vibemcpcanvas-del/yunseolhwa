"""Verification tests for Gate 2: Tunnel Checkpoint Server True ACKs & Integrity."""

import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
import pytest

from cloud.colab_tunnel_bridge import JaxTunnelTransferManager


@pytest.fixture
def test_env():
    d = tempfile.mkdtemp(prefix="test_tunnel_")
    data_tar = os.path.join(d, "upload_jax.tar")
    with open(data_tar, "wb") as f:
        f.write(b"mock_data")

    script_path = os.path.join(d, "job.py")
    with open(script_path, "wb") as f:
        f.write(b"print('job')")

    out_tar = os.path.join(d, "out.tar")
    extract_dir = os.path.join(d, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    manager = JaxTunnelTransferManager(
        data_tar_path=data_tar,
        script_path=script_path,
        output_checkpoints_tar=out_tar,
        extract_target_dir=extract_dir,
    )
    url = manager.start_server_only()

    yield {
        "dir": d,
        "manager": manager,
        "url": url,
        "extract_dir": extract_dir,
    }

    manager.stop()
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)


def _create_mock_checkpoint_tar(step_name="step_50", extra_content=b"weights_v1"):
    """Creates an in-memory valid tar archive containing a mock checkpoint folder."""
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w") as tar:
        # Add a file inside step folder
        ti = tarfile.TarInfo(name=f"{step_name}/params.msgpack")
        ti.size = len(extra_content)
        tar.addfile(ti, io.BytesIO(extra_content))

        # Add manifest.json
        manifest = json.dumps({"schema_version": 1, "update": 50}).encode("utf-8")
        ti_m = tarfile.TarInfo(name=f"{step_name}/manifest.json")
        ti_m.size = len(manifest)
        tar.addfile(ti_m, io.BytesIO(manifest))
    return bio.getvalue()


def test_corrupted_payload_and_sha256_mismatch(test_env):
    """Test 1: SHA256 mismatch or corrupted payload causes HTTP 400 and step is NOT verified."""
    url = test_env["url"]
    manager = test_env["manager"]

    tar_bytes = _create_mock_checkpoint_tar("step_50")
    bad_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"

    req = urllib.request.Request(
        f"{url}/upload_checkpoint?step=step_50&sha256={bad_sha256}",
        data=tar_bytes,
        method="POST",
        headers={"Content-Length": str(len(tar_bytes))}
    )

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)

    assert exc_info.value.code == 400
    assert not manager.is_step_verified(50)
    assert manager.get_last_verified_step() == 0


def test_idempotent_duplicate_upload(test_env):
    """Test 2: Sending identical valid checkpoint tar twice returns 200 idempotently."""
    url = test_env["url"]
    manager = test_env["manager"]

    tar_bytes = _create_mock_checkpoint_tar("step_50", extra_content=b"identical_payload")
    sha256 = hashlib.sha256(tar_bytes).hexdigest()

    req1 = urllib.request.Request(
        f"{url}/upload_checkpoint?step=step_50&sha256={sha256}",
        data=tar_bytes,
        method="POST",
        headers={"Content-Length": str(len(tar_bytes))}
    )

    # First upload: returns 200, adds to verified_steps
    with urllib.request.urlopen(req1) as resp:
        assert resp.status == 200
        assert resp.read() == b"CHECKPOINT_OK"

    assert manager.is_step_verified(50)
    assert manager.get_last_verified_step() == 50

    # Second upload of the identical payload: returns 200 idempotently
    req2 = urllib.request.Request(
        f"{url}/upload_checkpoint?step=step_50&sha256={sha256}",
        data=tar_bytes,
        method="POST",
        headers={"Content-Length": str(len(tar_bytes))}
    )
    with urllib.request.urlopen(req2) as resp:
        assert resp.status == 200
        assert resp.read() == b"CHECKPOINT_OK_IDEMPOTENT"

    assert manager.is_step_verified(50)


def test_conflicting_hash_returns_409(test_env):
    """Test 3: Sending different payload with same step identifier raises HTTP 409 Conflict."""
    url = test_env["url"]
    manager = test_env["manager"]

    tar_bytes_v1 = _create_mock_checkpoint_tar("step_50", extra_content=b"version_1")
    sha256_v1 = hashlib.sha256(tar_bytes_v1).hexdigest()

    req1 = urllib.request.Request(
        f"{url}/upload_checkpoint?step=step_50&sha256={sha256_v1}",
        data=tar_bytes_v1,
        method="POST",
        headers={"Content-Length": str(len(tar_bytes_v1))}
    )
    with urllib.request.urlopen(req1) as resp:
        assert resp.status == 200

    # Now attempt to upload different content under the same step_50
    tar_bytes_v2 = _create_mock_checkpoint_tar("step_50", extra_content=b"version_2_conflicting")
    sha256_v2 = hashlib.sha256(tar_bytes_v2).hexdigest()

    req2 = urllib.request.Request(
        f"{url}/upload_checkpoint?step=step_50&sha256={sha256_v2}",
        data=tar_bytes_v2,
        method="POST",
        headers={"Content-Length": str(len(tar_bytes_v2))}
    )

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req2)

    assert exc_info.value.code == 409

"""Unit tests for boxd_adk tools and plugin (no network access)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from boxd import AuthenticationError, BoxdError, NotFoundError
from boxd.errors import MachineFileNotFoundError
from boxd.models import Checkpoint, CreatedCheckpoint, ExecResult

import boxd_adk.tools as tools_module
from boxd_adk import BoxdPlugin


def completed(exit_code=0, stdout="", stderr=""):
    return ExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code)


def fake_machine(machine_id="m-1", status="running"):
    machine = MagicMock()
    machine.id = machine_id
    machine.status = status
    return machine


def make_plugin(**kwargs):
    with patch("boxd_adk.plugin.Boxd"):
        return BoxdPlugin(api_key="test-key", **kwargs)


def make_plugin_with_machine(**kwargs):
    plugin = make_plugin(**kwargs)
    plugin.get_machine = MagicMock(return_value=fake_machine())
    return plugin, plugin.client.machines


def tool_by_name(plugin, name):
    return next(t for t in plugin.get_tools() if t.name == name)


def run_tool(tool, args):
    return asyncio.run(tool.run_async(args=args, tool_context=MagicMock()))


# -- plugin: credentials and construction -------------------------------------


def test_api_key_required_outside_a_machine():
    with patch("boxd_adk.plugin.Boxd"), patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="BOXD_API_KEY"):
            BoxdPlugin()


def test_api_key_from_env():
    with (
        patch("boxd_adk.plugin.Boxd") as boxd_cls,
        patch.dict("os.environ", {"BOXD_API_KEY": "env-key"}, clear=True),
    ):
        plugin = BoxdPlugin()
        assert plugin.machine_name.startswith("adk-")
        assert boxd_cls.call_args.kwargs["api_key"] == "env-key"


def test_no_api_key_needed_inside_a_boxd_machine():
    # Inside a machine the SDK authenticates through the metadata adapter.
    with (
        patch("boxd_adk.plugin.Boxd") as boxd_cls,
        patch.dict("os.environ", {"BOXD_VM_ID": "vm-123"}, clear=True),
    ):
        BoxdPlugin()
        assert boxd_cls.call_args.kwargs["api_key"] is None


def test_passes_region_and_base_url_to_client():
    with patch("boxd_adk.plugin.Boxd") as boxd_cls:
        BoxdPlugin(api_key="k", region="fra", base_url="https://example:9443", client_timeout=5)
        kwargs = boxd_cls.call_args.kwargs
        assert kwargs["region"] == "fra"
        assert kwargs["base_url"] == "https://example:9443"
        assert kwargs["timeout"] == 5


def test_constructor_makes_no_api_calls():
    with patch("boxd_adk.plugin.Boxd") as boxd_cls:
        BoxdPlugin(api_key="k")
        assert boxd_cls.return_value.mock_calls == []


def test_given_client_is_used_and_not_closed():
    client = MagicMock()
    with patch.dict("os.environ", {}, clear=True):
        plugin = BoxdPlugin(client=client)  # no key needed: caller owns auth
    assert plugin.client is client
    plugin._destroy_on_close = False
    asyncio.run(plugin.close())
    client.close.assert_not_called()


def test_from_snapshot_and_image_are_exclusive():
    with pytest.raises(ValueError, match="from_snapshot or image"):
        make_plugin(from_snapshot="golden", image="ubuntu:24.04")


def test_named_machine_is_persistent_by_default():
    plugin = make_plugin(machine_name="my-project")
    assert plugin.machine_name == "my-project"
    assert plugin._destroy_on_close is False


def test_unnamed_machine_is_destroyed_by_default():
    plugin = make_plugin()
    assert plugin.machine_name.startswith("adk-")
    assert plugin._destroy_on_close is True


def test_destroy_on_close_override():
    assert make_plugin(machine_name="x", destroy_on_close=True)._destroy_on_close is True
    assert make_plugin(destroy_on_close=False)._destroy_on_close is False


# -- plugin: machine lifecycle -------------------------------------------------


def test_get_machine_reuses_existing():
    plugin = make_plugin(machine_name="existing")
    existing = fake_machine(status="hibernated")
    plugin.client.machines.get.return_value = existing
    assert plugin.get_machine() is existing
    plugin.client.machines.create.assert_not_called()
    # A sleeping machine is left for the next command to wake.
    plugin.client.machines.start.assert_not_called()
    # Cached on second call.
    plugin.get_machine()
    plugin.client.machines.get.assert_called_once_with("existing")


def test_get_machine_starts_a_stopped_machine():
    plugin = make_plugin(machine_name="stopped")
    plugin.client.machines.get.return_value = fake_machine("m-9", status="stopped")
    plugin.get_machine()
    plugin.client.machines.start.assert_called_once_with("m-9")
    plugin.client.machines.wait_until_ready.assert_called_once_with("m-9")


def test_get_machine_creates_when_missing():
    plugin = make_plugin(machine_name="fresh", vcpu=2, delete_after=3600)
    plugin.client.machines.get.side_effect = NotFoundError("nope")
    created = fake_machine("m-new")
    plugin.client.machines.create.return_value = created
    assert plugin.get_machine() is created
    args, kwargs = plugin.client.machines.create.call_args
    assert args == ("fresh",)
    assert kwargs == {
        "vcpu": 2,
        "delete_after": 3600,
        "labels": {"created-by": "google-adk"},
    }
    plugin.client.machines.wait_until_ready.assert_called_once()


def test_create_merges_user_labels_over_default():
    plugin = make_plugin(labels={"team": "ml", "created-by": "me"}, from_snapshot="golden")
    plugin.client.machines.get.side_effect = NotFoundError("nope")
    plugin.client.machines.create.return_value = fake_machine()
    plugin.get_machine()
    kwargs = plugin.client.machines.create.call_args.kwargs
    assert kwargs["labels"] == {"created-by": "me", "team": "ml"}
    assert kwargs["from_snapshot"] == "golden"
    assert "image" not in kwargs


def test_get_machine_does_not_create_when_name_is_in_another_region():
    plugin = make_plugin(machine_name="elsewhere")
    err = NotFoundError("there is a machine 'elsewhere' in nyc")
    err.elsewhere = [MagicMock()]
    plugin.client.machines.get.side_effect = err
    with pytest.raises(NotFoundError):
        plugin.get_machine()
    plugin.client.machines.create.assert_not_called()


def test_close_destroys_when_configured():
    plugin = make_plugin()  # unnamed -> destroy_on_close=True
    plugin._machine = fake_machine("m-7")
    asyncio.run(plugin.close())
    plugin.client.machines.delete.assert_called_once_with("m-7")
    plugin.client.close.assert_called_once()
    assert plugin._machine is None


def test_close_before_first_use_deletes_by_name():
    plugin = make_plugin()
    plugin.client.machines.delete.side_effect = NotFoundError("never created")
    asyncio.run(plugin.close())  # must not raise
    plugin.client.machines.delete.assert_called_once_with(plugin.machine_name)


def test_close_preserves_named_machine():
    plugin = make_plugin(machine_name="keep-me")
    asyncio.run(plugin.close())
    plugin.client.machines.delete.assert_not_called()
    plugin.client.close.assert_called_once()


def test_fork_returns_plugin_bound_to_the_copy():
    plugin = make_plugin(machine_name="base", delete_after=600)
    plugin._machine = fake_machine("m-base")
    plugin.client.machines.fork.return_value = fake_machine("m-fork")
    child = plugin.fork()
    args, kwargs = plugin.client.machines.fork.call_args
    assert args[0] == "m-base"
    assert args[1] == child.machine_name and child.machine_name.startswith("adk-")
    assert kwargs["labels"] == {"created-by": "google-adk"}
    assert kwargs["delete_after"] == 600
    assert child.get_machine().id == "m-fork"
    assert child.client is plugin.client
    assert child._destroy_on_close is True
    # Closing the fork deletes the fork only and leaves the shared client open.
    asyncio.run(child.close())
    plugin.client.machines.delete.assert_called_once_with("m-fork")
    plugin.client.close.assert_not_called()


def test_named_fork_is_kept():
    plugin = make_plugin(machine_name="base")
    plugin._machine = fake_machine("m-base")
    plugin.client.machines.fork.return_value = fake_machine("m-fork")
    child = plugin.fork("experiment")
    assert child.machine_name == "experiment"
    assert child._destroy_on_close is False
    assert "delete_after" not in plugin.client.machines.fork.call_args.kwargs


def test_save_snapshot_waits_until_ready(monkeypatch):
    monkeypatch.setattr("boxd_adk.plugin.time.sleep", lambda _: None)
    plugin, _machines = make_plugin_with_machine()
    plugin.client.snapshots.get.side_effect = [
        NotFoundError("not yet"),
        MagicMock(status="pending"),
        MagicMock(status="ready"),
    ]
    assert plugin.save_snapshot("golden") == "golden"
    plugin.client.snapshots.create.assert_called_once_with("m-1", "golden")
    assert plugin.client.snapshots.get.call_count == 3


def test_save_snapshot_without_waiting():
    plugin, _machines = make_plugin_with_machine()
    assert plugin.save_snapshot("golden", wait=False) == "golden"
    plugin.client.snapshots.get.assert_not_called()


def test_save_snapshot_failure_raises(monkeypatch):
    monkeypatch.setattr("boxd_adk.plugin.time.sleep", lambda _: None)
    plugin, _machines = make_plugin_with_machine()
    plugin.client.snapshots.get.return_value = MagicMock(status="failed")
    with pytest.raises(RuntimeError, match="failed"):
        plugin.save_snapshot("golden")


def test_get_tools_names():
    plugin = make_plugin()
    names = {t.name for t in plugin.get_tools()}
    assert names == {
        "execute_command_in_boxd",
        "execute_code_in_boxd",
        "write_file_to_boxd",
        "read_file_from_boxd",
        "create_boxd_checkpoint",
        "list_boxd_checkpoints",
        "restore_boxd_checkpoint",
    }


def test_on_tool_error_returns_structured_result_for_own_tools():
    plugin = make_plugin()
    tool = tool_by_name(plugin, "execute_command_in_boxd")
    result = asyncio.run(
        plugin.on_tool_error_callback(
            tool=tool, tool_args={}, tool_context=MagicMock(), error=RuntimeError("boom")
        )
    )
    assert result == {"success": False, "error": "boom"}


def test_on_tool_error_ignores_foreign_tools():
    plugin = make_plugin()
    foreign = MagicMock()
    foreign.name = "some_other_tool"
    result = asyncio.run(
        plugin.on_tool_error_callback(
            tool=foreign, tool_args={}, tool_context=MagicMock(), error=RuntimeError("boom")
        )
    )
    assert result is None


def test_on_tool_error_ignores_same_named_tool_of_another_plugin():
    # Two plugins (e.g. a machine and its fork) have tools with the same names.
    plugin, other = make_plugin(), make_plugin()
    result = asyncio.run(
        plugin.on_tool_error_callback(
            tool=tool_by_name(other, "execute_command_in_boxd"),
            tool_args={},
            tool_context=MagicMock(),
            error=RuntimeError("boom"),
        )
    )
    assert result is None


# -- execute_command / execute_code -------------------------------------------


def test_execute_command():
    plugin, machines = make_plugin_with_machine()
    machines.exec.return_value = completed(0, stdout="hello\n")
    result = run_tool(tool_by_name(plugin, "execute_command_in_boxd"), {"command": "echo hello"})
    assert result == {"success": True, "exit_code": 0, "stdout": "hello\n", "stderr": ""}
    args, kwargs = machines.exec.call_args
    assert args == ("m-1", "echo hello")
    assert kwargs == {"cwd": None, "timeout": 300.0}


def test_execute_command_passes_cwd_and_timeout():
    plugin, machines = make_plugin_with_machine()
    machines.exec.return_value = completed(0)
    run_tool(
        tool_by_name(plugin, "execute_command_in_boxd"),
        {"command": "make", "cwd": "/srv/app", "timeout": 30},
    )
    assert machines.exec.call_args.kwargs == {"cwd": "/srv/app", "timeout": 30.0}


def test_execute_command_failure_exit_code():
    plugin, machines = make_plugin_with_machine()
    machines.exec.return_value = completed(1, stderr="no such file\n")
    result = run_tool(tool_by_name(plugin, "execute_command_in_boxd"), {"command": "cat /nope"})
    assert result["success"] is False
    assert result["exit_code"] == 1
    assert "no such file" in result["stderr"]


def test_execute_command_truncates_huge_output():
    plugin, machines = make_plugin_with_machine()
    machines.exec.return_value = completed(0, stdout="x" * 150_000)
    result = run_tool(tool_by_name(plugin, "execute_command_in_boxd"), {"command": "yes"})
    assert len(result["stdout"]) < 101_000
    assert "truncated 50000 characters" in result["stdout"]


def test_execute_command_requires_command():
    plugin, machines = make_plugin_with_machine()
    result = run_tool(tool_by_name(plugin, "execute_command_in_boxd"), {})
    assert result["success"] is False
    machines.exec.assert_not_called()


def test_execute_command_timeout_is_reported():
    plugin, machines = make_plugin_with_machine()
    machines.exec.side_effect = BoxdError("exec timed out after 2s")
    result = run_tool(
        tool_by_name(plugin, "execute_command_in_boxd"), {"command": "sleep 9", "timeout": 2}
    )
    assert result["success"] is False
    assert result["timed_out"] is True
    assert "2s" in result["error"]


def test_execute_command_exception_becomes_error_dict():
    plugin, machines = make_plugin_with_machine()
    machines.exec.side_effect = RuntimeError("connection lost")
    result = run_tool(tool_by_name(plugin, "execute_command_in_boxd"), {"command": "ls"})
    assert result == {"success": False, "error": "connection lost"}


def test_authentication_error_points_at_the_api_key():
    plugin = make_plugin()
    plugin.get_machine = MagicMock(side_effect=AuthenticationError("API key not recognized"))
    result = run_tool(tool_by_name(plugin, "execute_command_in_boxd"), {"command": "ls"})
    assert result["success"] is False
    assert "BOXD_API_KEY" in result["error"]
    assert "API key not recognized" in result["error"]


def test_execute_code_python_default():
    plugin, machines = make_plugin_with_machine()
    machines.exec.return_value = completed(0, stdout="4\n")
    result = run_tool(tool_by_name(plugin, "execute_code_in_boxd"), {"code": "print(2+2)"})
    assert result["success"] is True
    assert result["language"] == "python"
    assert result["stdout"] == "4\n"
    args, _ = machines.exec.call_args
    assert args == ("m-1", ["python3", "-c", "print(2+2)"])


def test_execute_code_javascript():
    plugin, machines = make_plugin_with_machine()
    machines.exec.return_value = completed(0, stdout="hi\n")
    result = run_tool(
        tool_by_name(plugin, "execute_code_in_boxd"),
        {"code": "console.log('hi')", "language": "javascript"},
    )
    assert result["success"] is True
    args, _ = machines.exec.call_args
    assert args[1][:2] == ["node", "-e"]


def test_execute_code_rejects_unknown_language():
    plugin, machines = make_plugin_with_machine()
    result = run_tool(
        tool_by_name(plugin, "execute_code_in_boxd"), {"code": "x", "language": "cobol"}
    )
    assert result["success"] is False
    assert "unsupported language" in result["error"]
    machines.exec.assert_not_called()


# -- files ---------------------------------------------------------------------


def test_write_file_uses_fs_api():
    plugin, machines = make_plugin_with_machine()
    content = "print('hi')\nx = \"quoted\" é\n"
    result = run_tool(
        tool_by_name(plugin, "write_file_to_boxd"),
        {"path": "/app/main.py", "content": content},
    )
    assert result == {
        "success": True,
        "path": "/app/main.py",
        "bytes_written": len(content.encode()),
    }
    machines.fs.write.assert_called_once_with("m-1", "/app/main.py", content.encode())


def test_write_file_allows_empty_content():
    plugin, _machines = make_plugin_with_machine()
    result = run_tool(tool_by_name(plugin, "write_file_to_boxd"), {"path": "/a", "content": ""})
    assert result["success"] is True
    assert result["bytes_written"] == 0


def test_write_file_requires_path():
    plugin, machines = make_plugin_with_machine()
    result = run_tool(tool_by_name(plugin, "write_file_to_boxd"), {"content": "x"})
    assert result["success"] is False
    machines.fs.write.assert_not_called()


def test_read_file():
    plugin, machines = make_plugin_with_machine()
    machines.fs.read.return_value = "file data é".encode()
    result = run_tool(tool_by_name(plugin, "read_file_from_boxd"), {"path": "/app/out.txt"})
    assert result == {
        "success": True,
        "path": "/app/out.txt",
        "content": "file data é",
        "truncated": False,
    }
    machines.fs.read.assert_called_once_with("m-1", "/app/out.txt")


def test_read_missing_file_errors():
    plugin, machines = make_plugin_with_machine()
    machines.fs.read.side_effect = MachineFileNotFoundError(
        "/nope: No such file or directory", code="not_found"
    )
    result = run_tool(tool_by_name(plugin, "read_file_from_boxd"), {"path": "/nope"})
    assert result["success"] is False
    assert "No such file" in result["error"]


def test_read_file_truncates():
    plugin, machines = make_plugin_with_machine()
    machines.fs.read.return_value = b"y" * 100_001
    result = run_tool(tool_by_name(plugin, "read_file_from_boxd"), {"path": "/big"})
    assert result["truncated"] is True
    assert result["content"].endswith("[truncated]")


# -- checkpoints ----------------------------------------------------------------


def _checkpoint(name, minutes_ago, status="ready"):
    return Checkpoint(
        id=f"ckpt_{name}",
        name=name,
        status=status,
        size_bytes=1,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        created_by="me",
        available=status == "ready",
    )


def test_create_checkpoint_waits_until_ready(monkeypatch):
    monkeypatch.setattr(tools_module, "_CHECKPOINT_POLL_INTERVAL", 0)
    plugin, machines = make_plugin_with_machine()
    machines.checkpoints.create.return_value = CreatedCheckpoint(
        id="ckpt_1", name="before-migration", status="pending"
    )
    machines.checkpoints.list.side_effect = [
        [_checkpoint("before-migration", 0, "pending")],
        [_checkpoint("older", 9), _checkpoint("before-migration", 0, "ready")],
    ]
    result = run_tool(tool_by_name(plugin, "create_boxd_checkpoint"), {"name": "before-migration"})
    assert result == {"success": True, "checkpoint_name": "before-migration", "status": "ready"}
    machines.checkpoints.create.assert_called_once_with("m-1", "before-migration")
    assert machines.checkpoints.list.call_count == 2


def test_create_checkpoint_default_name():
    plugin, machines = make_plugin_with_machine()
    machines.checkpoints.create.side_effect = lambda mid, name: CreatedCheckpoint(
        id="ckpt_1", name=name, status="ready"
    )
    result = run_tool(tool_by_name(plugin, "create_boxd_checkpoint"), {})
    assert result["success"] is True
    assert result["checkpoint_name"].startswith("adk-")


def test_create_checkpoint_surfaces_failure(monkeypatch):
    monkeypatch.setattr(tools_module, "_CHECKPOINT_POLL_INTERVAL", 0)
    plugin, machines = make_plugin_with_machine()
    machines.checkpoints.create.return_value = CreatedCheckpoint(
        id="ckpt_1", name="cp", status="pending"
    )
    machines.checkpoints.list.return_value = [_checkpoint("cp", 0, "failed")]
    result = run_tool(tool_by_name(plugin, "create_boxd_checkpoint"), {"name": "cp"})
    assert result["success"] is False
    assert "failed" in result["error"]


def test_list_checkpoints_newest_first():
    plugin, machines = make_plugin_with_machine()
    machines.checkpoints.list.return_value = [
        _checkpoint("first", 90),
        _checkpoint("third", 10, "pending"),
        _checkpoint("second", 60),
    ]
    result = run_tool(tool_by_name(plugin, "list_boxd_checkpoints"), {})
    assert result["success"] is True
    assert [c["checkpoint_name"] for c in result["checkpoints"]] == ["third", "second", "first"]
    assert result["checkpoints"][0]["restorable"] is False
    assert result["checkpoints"][1]["restorable"] is True


def test_restore_requires_confirmation():
    plugin, machines = make_plugin_with_machine()
    result = run_tool(tool_by_name(plugin, "restore_boxd_checkpoint"), {"checkpoint_name": "cp-1"})
    assert result["success"] is False
    assert "confirm" in result["error"]
    machines.checkpoints.restore.assert_not_called()


def test_restore_with_confirmation():
    plugin, machines = make_plugin_with_machine()
    result = run_tool(
        tool_by_name(plugin, "restore_boxd_checkpoint"),
        {"checkpoint_name": "cp-1", "confirm": True},
    )
    assert result == {"success": True, "checkpoint_name": "cp-1", "restored": True}
    machines.checkpoints.restore.assert_called_once_with("m-1", "cp-1")


def test_restore_confirm_must_be_boolean_true():
    plugin, machines = make_plugin_with_machine()
    result = run_tool(
        tool_by_name(plugin, "restore_boxd_checkpoint"),
        {"checkpoint_name": "cp-1", "confirm": "yes"},
    )
    assert result["success"] is False
    machines.checkpoints.restore.assert_not_called()


# -- declarations -----------------------------------------------------------------


def test_all_tools_have_declarations():
    plugin = make_plugin()
    for tool in plugin.get_tools():
        decl = tool._get_declaration()
        assert decl is not None
        assert decl.name == tool.name
        assert decl.description

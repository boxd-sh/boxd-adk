"""Live tests against real boxd machines.

Skipped unless BOXD_ADK_LIVE=1 and BOXD_API_KEY are set. Every machine they
create is named ``t-adk-...``, labelled ``overnight=adk``, has a 30 minute
delete-after timer as a safety net, and is deleted at the end.

    BOXD_ADK_LIVE=1 BOXD_API_KEY=... pytest tests/test_live.py -v

The agent test also needs ANTHROPIC_API_KEY (it drives Claude through ADK's
LiteLLM support; set BOXD_ADK_TEST_MODEL to use another model).
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid

import pytest
from boxd import Boxd, NotFoundError

from boxd_adk import BoxdPlugin

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("BOXD_ADK_LIVE") != "1" or not os.environ.get("BOXD_API_KEY"),
        reason="live test: set BOXD_ADK_LIVE=1 and BOXD_API_KEY",
    ),
]

LABELS = {"overnight": "adk"}
SAFETY_NET = 1800  # seconds


def unique(prefix: str) -> str:
    return f"t-adk-{prefix}-{uuid.uuid4().hex[:6]}"


def gone(client, name: str, wait: float = 60.0) -> bool:
    """True once the machine no longer exists (deletes finish asynchronously)."""
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            client.machines.get(name)
        except NotFoundError:
            return True
        time.sleep(1)
    return False


def call(plugin: BoxdPlugin, tool_name: str, **args):
    tool = next(t for t in plugin.get_tools() if t.name == tool_name)
    return asyncio.run(tool.run_async(args=args, tool_context=None))


@pytest.fixture
def plugin():
    p = BoxdPlugin(
        machine_name=unique("live"),
        labels=LABELS,
        delete_after=SAFETY_NET,
        destroy_on_close=True,
    )
    yield p
    asyncio.run(p.close())


def test_commands_files_and_errors(plugin):
    started = time.monotonic()
    r = call(plugin, "execute_command_in_boxd", command="echo out; echo err >&2; exit 3")
    print(f"first command (incl. machine create) took {time.monotonic() - started:.2f}s")
    assert r == {"success": False, "exit_code": 3, "stdout": "out\n", "stderr": "err\n"}

    machine = plugin.client.machines.get(plugin.machine_name)
    assert machine.labels["overnight"] == "adk"
    assert machine.labels["created-by"] == "google-adk"
    assert machine.delete_at is not None

    r = call(plugin, "execute_command_in_boxd", command="pwd", cwd="/tmp")
    assert r["stdout"] == "/tmp\n"
    r = call(plugin, "execute_command_in_boxd", command="pwd", cwd="/does/not/exist")
    assert r["success"] is False

    r = call(plugin, "execute_command_in_boxd", command="sleep 20", timeout=2)
    assert r["success"] is False and r["timed_out"] is True

    r = call(plugin, "execute_code_in_boxd", code="import sys; print(sys.version_info[0], 'é')")
    assert r["success"] and r["stdout"] == "3 é\n" and r["language"] == "python"
    r = call(plugin, "execute_code_in_boxd", code="console.log(6*7)", language="javascript")
    assert r["stdout"] == "42\n"
    r = call(plugin, "execute_code_in_boxd", code="echo \"$((1+1)) 'quoted'\"", language="bash")
    assert r["stdout"] == "2 'quoted'\n"

    content = "line 1\nquotes: ' \" `$(whoami)`\nunicode: héllo ✓\n"
    r = call(plugin, "write_file_to_boxd", path="/home/boxd/a/b/c.txt", content=content)
    assert r == {
        "success": True,
        "path": "/home/boxd/a/b/c.txt",
        "bytes_written": len(content.encode()),
    }
    r = call(plugin, "read_file_from_boxd", path="/home/boxd/a/b/c.txt")
    assert r["content"] == content and r["truncated"] is False
    r = call(
        plugin, "execute_command_in_boxd", command="cat a/b/c.txt | wc -l; stat -c %U a/b/c.txt"
    )
    assert r["stdout"] == "3\nboxd\n"

    r = call(plugin, "write_file_to_boxd", path="rel.txt", content="relative")
    assert r["success"]
    assert call(plugin, "read_file_from_boxd", path="/home/boxd/rel.txt")["content"] == "relative"

    big = "z" * 600_000  # well past a command line's size limit
    assert call(plugin, "write_file_to_boxd", path="/tmp/big.txt", content=big)["success"]
    r = call(plugin, "execute_command_in_boxd", command="wc -c < /tmp/big.txt")
    assert r["stdout"].strip() == "600000"
    assert call(plugin, "read_file_from_boxd", path="/tmp/big.txt")["truncated"] is True

    r = call(plugin, "read_file_from_boxd", path="/nope.txt")
    assert r["success"] is False and "No such file" in r["error"]
    r = call(plugin, "write_file_to_boxd", path="/root/x.txt", content="x")
    assert r["success"] is False and "Permission denied" in r["error"]


def test_checkpoint_restore_rolls_back(plugin):
    call(plugin, "write_file_to_boxd", path="/home/boxd/state.txt", content="v1")
    r = call(plugin, "create_boxd_checkpoint", name="before-experiment")
    assert r == {"success": True, "checkpoint_name": "before-experiment", "status": "ready"}
    call(plugin, "write_file_to_boxd", path="/home/boxd/state.txt", content="broken")

    listed = call(plugin, "list_boxd_checkpoints")["checkpoints"]
    assert listed[0]["checkpoint_name"] == "before-experiment"
    assert listed[0]["restorable"] is True

    r = call(plugin, "restore_boxd_checkpoint", checkpoint_name="before-experiment")
    assert r["success"] is False  # no confirm
    assert call(plugin, "read_file_from_boxd", path="state.txt")["content"] == "broken"

    r = call(plugin, "restore_boxd_checkpoint", checkpoint_name="before-experiment", confirm=True)
    assert r == {"success": True, "checkpoint_name": "before-experiment", "restored": True}
    assert call(plugin, "read_file_from_boxd", path="state.txt")["content"] == "v1"

    r = call(plugin, "restore_boxd_checkpoint", checkpoint_name="no-such-checkpoint", confirm=True)
    assert r["success"] is False and r["error"]

    r = call(plugin, "create_boxd_checkpoint")
    assert r["success"] and r["checkpoint_name"].startswith("adk-")


def test_named_machine_persists_and_wakes_from_hibernation():
    name = unique("persist")
    first = BoxdPlugin(machine_name=name, labels=LABELS, delete_after=SAFETY_NET)
    second = None
    try:
        call(first, "write_file_to_boxd", path="notes.txt", content="from session 1")
        asyncio.run(first.close())  # named: kept
        second = BoxdPlugin(machine_name=name)
        machine = second.client.machines.get(name)
        assert machine.status == "running"
        second.client.machines.hibernate(machine.id)
        for _ in range(60):
            if second.client.machines.get(name).status in ("suspended", "hibernated"):
                break
            time.sleep(1)
        print("status before tool call:", second.client.machines.get(name).status)
        started = time.monotonic()
        r = call(second, "read_file_from_boxd", path="notes.txt")
        print(f"read after hibernate took {time.monotonic() - started:.2f}s")
        assert r["content"] == "from session 1"
        assert second.client.machines.get(name).status == "running"
    finally:
        if second is None:
            second = BoxdPlugin(machine_name=name)
        second.destroy_machine()
        asyncio.run(second.close())


def test_fork_and_snapshot(plugin):
    call(plugin, "write_file_to_boxd", path="base.txt", content="base")
    # A process started before the fork keeps running in the copy.
    call(plugin, "execute_command_in_boxd", command="nohup sleep 600 >/dev/null 2>&1 &")

    child = plugin.fork(unique("fork"), destroy_on_close=True)
    try:
        assert child.machine_name != plugin.machine_name
        assert child.client.machines.get(child.machine_name).labels["overnight"] == "adk"
        assert call(child, "read_file_from_boxd", path="base.txt")["content"] == "base"
        assert call(child, "execute_command_in_boxd", command="pgrep -x sleep")["success"]
        call(child, "write_file_to_boxd", path="base.txt", content="changed in fork")
        assert call(plugin, "read_file_from_boxd", path="base.txt")["content"] == "base"
    finally:
        asyncio.run(child.close())
    assert gone(plugin.client, child.machine_name)

    snap = unique("snap")
    restored = None
    try:
        started = time.monotonic()
        assert plugin.save_snapshot(snap) == snap
        print(f"snapshot ready after {time.monotonic() - started:.1f}s")
        assert plugin.client.snapshots.get(snap).status == "ready"
        restored = BoxdPlugin(
            machine_name=unique("from-snap"),
            from_snapshot=snap,
            labels=LABELS,
            delete_after=SAFETY_NET,
            destroy_on_close=True,
        )
        assert call(restored, "read_file_from_boxd", path="base.txt")["content"] == "base"
        m = restored.client.machines.get(restored.machine_name)
        assert m.source is not None and m.source.kind == "snapshot"
    finally:
        if restored is not None:
            asyncio.run(restored.close())
        plugin.client.snapshots.delete(snap)


def test_wrong_api_key_gives_clear_error(monkeypatch):
    monkeypatch.setenv("BOXD_API_KEY", "not-a-real-boxd-key")
    p = BoxdPlugin(machine_name=unique("badkey"))
    try:
        r = call(p, "execute_command_in_boxd", command="true")
        assert r["success"] is False
        assert "BOXD_API_KEY" in r["error"]
        print("error shown to the model:", r["error"])
    finally:
        asyncio.run(p.close())


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_real_adk_agent_uses_the_machine():
    from google.adk.agents import Agent
    from google.adk.apps import App
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    model = os.environ.get("BOXD_ADK_TEST_MODEL", "anthropic/claude-sonnet-4-5")
    plugin = BoxdPlugin(
        machine_name=unique("agent"), labels=LABELS, delete_after=SAFETY_NET, destroy_on_close=True
    )
    agent = Agent(
        model=model,
        name="boxd_agent",
        instruction="Run all code and commands in the boxd machine using your tools.",
        tools=plugin.get_tools(),
    )
    prompt = (
        "Write a Python script /home/boxd/fib.py that prints the first 15 Fibonacci "
        "numbers separated by spaces, run it, and tell me the last number."
    )

    async def run():
        calls = []
        final = ""
        app = App(name="live_test", root_agent=agent, plugins=[plugin])
        async with InMemoryRunner(app=app) as runner:
            session = await runner.session_service.create_session(
                app_name=runner.app_name, user_id="test"
            )
            async for event in runner.run_async(
                user_id="test",
                session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
            ):
                for call_ in event.get_function_calls():
                    calls.append(call_.name)
                if event.is_final_response() and event.content and event.content.parts:
                    final = "".join(p.text or "" for p in event.content.parts)
            # Check the machine before the runner closes the plugin.
            out = await asyncio.to_thread(
                plugin.client.machines.exec, plugin.machine_id, "python3 /home/boxd/fib.py"
            )
        return calls, final, out

    calls, final, out = asyncio.run(run())
    print("tool calls:", calls)
    print("final answer:", final)
    assert "write_file_to_boxd" in calls or "execute_command_in_boxd" in calls
    assert out.stdout.split()[-1] == "377"
    assert "377" in final
    # Closing the runner closed the plugin, which deleted the machine
    # (destroy_on_close=True) and closed its client.
    with Boxd() as boxd:
        assert gone(boxd, plugin.machine_name)

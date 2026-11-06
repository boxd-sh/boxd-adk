"""ADK tools backed by a boxd machine.

Every tool shares the plugin's single machine. Tool bodies are synchronous
(they use the synchronous ``boxd.Boxd`` client) and are executed on a worker
thread via ``asyncio.to_thread`` so they never block the agent's event loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from boxd import AuthenticationError, BoxdError
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

if TYPE_CHECKING:
    from .plugin import BoxdPlugin

logger = logging.getLogger("boxd_adk")

_DEFAULT_EXEC_TIMEOUT = 300.0

# Output and file content handed back to the model is capped so one noisy
# command can't flood the context window.
_MAX_OUTPUT_CHARS = 100_000

# How long create_boxd_checkpoint waits for a new checkpoint to become
# restorable before returning it as still pending.
_CHECKPOINT_READY_TIMEOUT = 120.0
_CHECKPOINT_POLL_INTERVAL = 1.0

_LANGUAGE_RUNNERS = {
    "python": ("python3", "-c"),
    "javascript": ("node", "-e"),
    "bash": ("bash", "-c"),
}


def _truncate(text: str | None, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if not text:
        return ""
    if len(text) > limit:
        return text[:limit] + f"\n... [truncated {len(text) - limit} characters]"
    return text


def _timeout(args: dict[str, Any]) -> float:
    return float(args.get("timeout") or _DEFAULT_EXEC_TIMEOUT)


def _exec_result(plugin: BoxdPlugin, command, *, cwd=None, timeout: float) -> dict[str, Any]:
    machine_id = plugin.machine_id
    try:
        result = plugin.client.machines.exec(machine_id, command, cwd=cwd, timeout=timeout)
    except BoxdError as e:
        if "timed out" in str(e):
            return {
                "success": False,
                "timed_out": True,
                "error": (
                    f"command did not finish within {timeout:g}s and was stopped. "
                    "Pass a larger timeout, or run long jobs in the background "
                    "(e.g. `nohup cmd > /tmp/log 2>&1 &`) and poll the log."
                ),
            }
        raise
    return {
        "success": result.exit_code == 0,
        "exit_code": result.exit_code,
        "stdout": _truncate(result.stdout),
        "stderr": _truncate(result.stderr),
    }


def _checkpoint_name() -> str:
    return "adk-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


class _BoxdTool(BaseTool):
    """Base class wiring a tool to the plugin's shared machine."""

    def __init__(self, plugin: BoxdPlugin, *, name: str, description: str):
        super().__init__(name=name, description=description)
        self._plugin = plugin

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(self._run, args)
        except AuthenticationError as e:
            logger.error("%s failed: %s", self.name, e)
            return {
                "success": False,
                "error": f"boxd rejected the credentials ({e}). Check BOXD_API_KEY.",
            }
        except Exception as e:  # surface failures to the model, don't crash the run
            logger.error("%s failed: %s", self.name, e)
            return {"success": False, "error": str(e)}

    def _run(self, args: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class ExecuteCommandTool(_BoxdTool):
    """Run a shell command inside the machine."""

    def __init__(self, plugin: BoxdPlugin):
        super().__init__(
            plugin,
            name="execute_command_in_boxd",
            description=(
                "Execute a shell command inside the boxd machine, a full Ubuntu "
                "Linux computer with sudo. Installed packages, files, and "
                "background processes persist between calls (and between "
                "sessions for named machines). Returns stdout, stderr, and the "
                "exit code."
            ),
        )

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "command": types.Schema(
                        type=types.Type.STRING,
                        description="The shell command to execute.",
                    ),
                    "cwd": types.Schema(
                        type=types.Type.STRING,
                        description="Working directory for the command (default /home/boxd).",
                    ),
                    "timeout": types.Schema(
                        type=types.Type.NUMBER,
                        description="Timeout in seconds (default 300).",
                    ),
                },
                required=["command"],
            ),
        )

    def _run(self, args: dict[str, Any]) -> dict[str, Any]:
        command = args.get("command", "")
        if not command:
            return {"success": False, "error": "command is required"}
        return _exec_result(
            self._plugin, command, cwd=args.get("cwd") or None, timeout=_timeout(args)
        )


class ExecuteCodeTool(_BoxdTool):
    """Run a code snippet inside the machine."""

    def __init__(self, plugin: BoxdPlugin):
        super().__init__(
            plugin,
            name="execute_code_in_boxd",
            description=(
                "Execute a code snippet inside the boxd machine. Supports python "
                "(python3), javascript (node), and bash. For multi-file programs, "
                "write files with write_file_to_boxd and run them with "
                "execute_command_in_boxd instead."
            ),
        )

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "code": types.Schema(
                        type=types.Type.STRING,
                        description="The code snippet to execute.",
                    ),
                    "language": types.Schema(
                        type=types.Type.STRING,
                        description="Language to run the snippet with (default python).",
                        enum=sorted(_LANGUAGE_RUNNERS),
                    ),
                    "timeout": types.Schema(
                        type=types.Type.NUMBER,
                        description="Timeout in seconds (default 300).",
                    ),
                },
                required=["code"],
            ),
        )

    def _run(self, args: dict[str, Any]) -> dict[str, Any]:
        code = args.get("code", "")
        if not code:
            return {"success": False, "error": "code is required"}
        language = (args.get("language") or "python").lower()
        runner = _LANGUAGE_RUNNERS.get(language)
        if runner is None:
            return {
                "success": False,
                "error": (
                    f"unsupported language {language!r}; use one of {sorted(_LANGUAGE_RUNNERS)}"
                ),
            }
        # A list is shell-quoted by the SDK, so the snippet reaches the
        # interpreter byte for byte.
        result = _exec_result(self._plugin, [*runner, code], timeout=_timeout(args))
        if "exit_code" in result:
            result = {"success": result["success"], "language": language, **result}
        return result


class WriteFileTool(_BoxdTool):
    """Write a file inside the machine."""

    def __init__(self, plugin: BoxdPlugin):
        super().__init__(
            plugin,
            name="write_file_to_boxd",
            description=(
                "Write text content to a file inside the boxd machine. Parent "
                "directories are created automatically and an existing file is "
                "replaced. Relative paths start in /home/boxd. Files persist "
                "for the lifetime of the machine."
            ),
        )

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path": types.Schema(
                        type=types.Type.STRING,
                        description="Path of the file inside the machine.",
                    ),
                    "content": types.Schema(
                        type=types.Type.STRING,
                        description="Text content to write.",
                    ),
                },
                required=["path", "content"],
            ),
        )

    def _run(self, args: dict[str, Any]) -> dict[str, Any]:
        path = args.get("path", "")
        content = args.get("content")
        if not path or content is None:
            return {"success": False, "error": "path and content are required"}
        data = content.encode("utf-8")
        # machines.fs writes as the machine's user, creates missing parents and
        # replaces the file atomically. It works on the machine's own disk, so
        # what it writes is captured by checkpoints like anything else.
        self._plugin.client.machines.fs.write(self._plugin.machine_id, path, data)
        return {"success": True, "path": path, "bytes_written": len(data)}


class ReadFileTool(_BoxdTool):
    """Read a file from the machine."""

    def __init__(self, plugin: BoxdPlugin):
        super().__init__(
            plugin,
            name="read_file_from_boxd",
            description=(
                "Read a text file from the boxd machine and return its content. "
                "Relative paths start in /home/boxd."
            ),
        )

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path": types.Schema(
                        type=types.Type.STRING,
                        description="Path of the file inside the machine.",
                    ),
                },
                required=["path"],
            ),
        )

    def _run(self, args: dict[str, Any]) -> dict[str, Any]:
        path = args.get("path", "")
        if not path:
            return {"success": False, "error": "path is required"}
        try:
            raw = self._plugin.client.machines.fs.read(self._plugin.machine_id, path)
        except OSError as e:  # the SDK's filesystem errors are OSErrors too
            return {"success": False, "error": str(e)}
        text = raw.decode("utf-8", errors="replace")
        truncated = len(text) > _MAX_OUTPUT_CHARS
        return {
            "success": True,
            "path": path,
            "content": text[:_MAX_OUTPUT_CHARS] + ("\n... [truncated]" if truncated else ""),
            "truncated": truncated,
        }


class CreateCheckpointTool(_BoxdTool):
    """Checkpoint the machine's full state."""

    def __init__(self, plugin: BoxdPlugin):
        super().__init__(
            plugin,
            name="create_boxd_checkpoint",
            description=(
                "Create a checkpoint of the boxd machine's entire state (disk, "
                "memory, and running processes). Use before risky operations - "
                "package upgrades, migrations, bulk edits - so the machine can "
                "be rolled back with restore_boxd_checkpoint."
            ),
        )

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "name": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "Short name saying why this checkpoint is taken, e.g. "
                            "'before-db-migration'. Letters, digits, '-' and '_'. "
                            "Defaults to a timestamp."
                        ),
                    ),
                },
            ),
        )

    def _run(self, args: dict[str, Any]) -> dict[str, Any]:
        name = args.get("name") or _checkpoint_name()
        client = self._plugin.client
        machine_id = self._plugin.machine_id
        created = client.machines.checkpoints.create(machine_id, name)
        # A new checkpoint is "pending" until its artifact lands, and only a
        # "ready" one can be restored. Wait, so a restore right after works.
        status = created.status
        deadline = time.monotonic() + _CHECKPOINT_READY_TIMEOUT
        while status == "pending" and time.monotonic() < deadline:
            time.sleep(_CHECKPOINT_POLL_INTERVAL)
            found = next(
                (c for c in client.machines.checkpoints.list(machine_id) if c.name == created.name),
                None,
            )
            status = found.status if found else status
        if status == "failed":
            return {"success": False, "error": f"checkpoint {created.name!r} failed"}
        return {"success": True, "checkpoint_name": created.name, "status": status}


class ListCheckpointsTool(_BoxdTool):
    """List the machine's checkpoints."""

    def __init__(self, plugin: BoxdPlugin):
        super().__init__(
            plugin,
            name="list_boxd_checkpoints",
            description=(
                "List the boxd machine's checkpoints (name, creation time, status), "
                "newest first. Use to find the checkpoint_name for a restore."
            ),
        )

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        )

    def _run(self, args: dict[str, Any]) -> dict[str, Any]:
        checkpoints = self._plugin.client.machines.checkpoints.list(self._plugin.machine_id)
        epoch = datetime.min.replace(tzinfo=timezone.utc)
        checkpoints.sort(key=lambda c: c.created_at or epoch, reverse=True)
        return {
            "success": True,
            "checkpoints": [
                {
                    "checkpoint_name": c.name,
                    "create_time": c.created_at.isoformat() if c.created_at else None,
                    "status": c.status,
                    "restorable": c.available,
                }
                for c in checkpoints
            ],
        }


class RestoreCheckpointTool(_BoxdTool):
    """Roll the machine back to a checkpoint. Destructive."""

    def __init__(self, plugin: BoxdPlugin):
        super().__init__(
            plugin,
            name="restore_boxd_checkpoint",
            description=(
                "Restore the boxd machine to a previous checkpoint. DESTRUCTIVE: "
                "this rewinds the entire machine (disk, memory and running "
                "processes) and permanently discards all changes made after the "
                "checkpoint. Confirm with the user before calling, and pass "
                "confirm=true only after they agree."
            ),
        )

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "checkpoint_name": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "Name of the checkpoint to restore (from list_boxd_checkpoints)."
                        ),
                    ),
                    "confirm": types.Schema(
                        type=types.Type.BOOLEAN,
                        description=(
                            "Must be true. Set only after the user has explicitly "
                            "confirmed they accept discarding state newer than the "
                            "checkpoint."
                        ),
                    ),
                },
                required=["checkpoint_name", "confirm"],
            ),
        )

    def _run(self, args: dict[str, Any]) -> dict[str, Any]:
        checkpoint_name = args.get("checkpoint_name", "")
        if not checkpoint_name:
            return {"success": False, "error": "checkpoint_name is required"}
        if args.get("confirm") is not True:
            return {
                "success": False,
                "error": (
                    "Restore not performed: restoring discards all changes made "
                    "after the checkpoint. Ask the user to confirm, then retry "
                    "with confirm=true."
                ),
            }
        # Returns once the machine runs again from the checkpoint.
        self._plugin.client.machines.checkpoints.restore(self._plugin.machine_id, checkpoint_name)
        return {"success": True, "checkpoint_name": checkpoint_name, "restored": True}

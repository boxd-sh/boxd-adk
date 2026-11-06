"""BoxdPlugin: Google ADK plugin for boxd machines."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from collections.abc import Mapping
from typing import Any

from boxd import Boxd, Machine, NotFoundError
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from .tools import (
    CreateCheckpointTool,
    ExecuteCodeTool,
    ExecuteCommandTool,
    ListCheckpointsTool,
    ReadFileTool,
    RestoreCheckpointTool,
    WriteFileTool,
)

logger = logging.getLogger("boxd_adk")

# Label put on every machine this plugin creates, so machines made by ADK
# agents can be found (`boxd machine list --label created-by=google-adk`).
# A `created-by` key in the user's own labels wins.
_CREATED_BY_LABEL = ("created-by", "google-adk")

# Environment variables that mean "running inside a boxd machine", where the
# SDK authenticates on its own through the machine's metadata adapter.
_IN_MACHINE_HINTS = ("BOXD_VM_ID", "BOXD_VM_NAME")

_SIGNUP_URL = "https://boxd.sh/?ref=boxd-adk"


class BoxdPlugin(BasePlugin):
    """Manages a boxd machine and exposes it to an ADK agent as tools.

    A boxd machine is a full Ubuntu computer (a KVM microVM) with a
    persistent disk. It boots in about a second, keeps its files, packages
    and running processes between agent sessions, and goes to sleep when
    idle; the next command wakes it.

    Two usage modes:

    * ``BoxdPlugin(machine_name="my-project")`` - get-or-create a named,
      persistent machine. The environment survives across sessions and is
      never deleted automatically.
    * ``BoxdPlugin()`` - create an ``adk-`` prefixed machine for this plugin
      instance and delete it when the plugin is closed.

    The machine is created lazily on first tool use, so constructing the
    plugin (and your agent) does not require network access.
    """

    def __init__(
        self,
        api_key: str | None = None,
        machine_name: str | None = None,
        *,
        plugin_name: str = "boxd_plugin",
        region: str | None = None,
        base_url: str | None = None,
        from_snapshot: str | None = None,
        image: str | None = None,
        vcpu: int | None = None,
        labels: Mapping[str, str] | None = None,
        delete_after: int | None = None,
        destroy_on_close: bool | None = None,
        client_timeout: float = 60.0,
        client: Boxd | None = None,
    ):
        """Initialize the plugin.

        Args:
            api_key: boxd API key. Falls back to the ``BOXD_API_KEY``
                environment variable. Inside a boxd machine it may be omitted:
                the SDK then authenticates as that machine's owner.
            machine_name: Name of the machine to use. If it exists it is
                reused (with all of its state); otherwise it is created.
                When omitted, a random ``adk-`` prefixed name is generated.
            plugin_name: ADK plugin identifier.
            region: boxd region for the machine (e.g. ``"fra"``). Defaults to
                ``BOXD_REGION``, then the region the API endpoint serves.
            base_url: boxd API endpoint. Defaults to ``BOXD_BASE_URL``, then
                the public boxd cluster.
            from_snapshot: Create the machine from this snapshot instead of
                the default image (only used when the machine is created).
            image: Image to boot a new machine from (only used when the
                machine is created). Defaults to the cluster's image.
            vcpu: Size of a new machine: 1, 2, 4 or 8 vCPUs, with 4 GB of
                memory per vCPU. Defaults to the organization's default size.
            labels: Labels for a new machine, merged over
                ``created-by=google-adk``.
            delete_after: Seconds after which boxd deletes a machine this
                plugin created, even if ``close()`` is never called. A safety
                net for crashed or killed processes. ``None`` sets no timer.
            destroy_on_close: Whether ``close()`` deletes the machine.
                Defaults to True for auto-generated names and False for
                user-supplied names (persistent environments).
            client_timeout: Timeout in seconds for boxd API calls (command
                execution has its own per-call timeout).
            client: An existing ``boxd.Boxd`` client to use instead of
                creating one. The plugin does not close a client it was given.
        """
        super().__init__(name=plugin_name)
        api_key = api_key or os.environ.get("BOXD_API_KEY") or None
        if (
            client is None
            and not api_key
            and not os.environ.get("BOXD_TOKEN")
            and not any(os.environ.get(v) for v in _IN_MACHINE_HINTS)
        ):
            raise ValueError(
                "A boxd API key is required. Pass api_key=... or set the "
                "BOXD_API_KEY environment variable. Create one in the boxd "
                f"console ({_SIGNUP_URL}) or with `boxd auth keys create <name>`."
            )
        if from_snapshot and image:
            raise ValueError("Pass either from_snapshot or image, not both.")

        if machine_name:
            self.machine_name = machine_name
            self._destroy_on_close = (
                bool(destroy_on_close) if destroy_on_close is not None else False
            )
        else:
            self.machine_name = f"adk-{uuid.uuid4().hex[:8]}"
            self._destroy_on_close = True if destroy_on_close is None else bool(destroy_on_close)

        self._create_options: dict[str, Any] = {
            "from_snapshot": from_snapshot,
            "image": image,
            "vcpu": vcpu,
            "labels": {_CREATED_BY_LABEL[0]: _CREATED_BY_LABEL[1], **dict(labels or {})},
            "delete_after": delete_after,
        }
        self._owns_client = client is None
        self._client = client or Boxd(
            api_key=api_key, base_url=base_url, region=region, timeout=client_timeout
        )
        self._machine: Machine | None = None
        self._lock = threading.Lock()
        self._tools = [
            ExecuteCommandTool(self),
            ExecuteCodeTool(self),
            WriteFileTool(self),
            ReadFileTool(self),
            CreateCheckpointTool(self),
            ListCheckpointsTool(self),
            RestoreCheckpointTool(self),
        ]

    # -- machine lifecycle --------------------------------------------------

    @property
    def client(self) -> Boxd:
        """The underlying boxd SDK client, for anything the tools don't cover."""
        return self._client

    def get_machine(self) -> Machine:
        """Return the managed machine, creating it on first use.

        Reuses the machine named ``machine_name`` when it exists; otherwise
        creates it and waits until it accepts commands. Thread-safe (tool
        calls run in worker threads).

        A machine that is asleep (standby or hibernated) is not woken here:
        boxd wakes it on the next command. A stopped machine is started.
        """
        with self._lock:
            if self._machine is None:
                try:
                    machine = self._client.machines.get(self.machine_name)
                    logger.info("Reusing existing machine %r", self.machine_name)
                    if machine.status == "stopped":
                        self._client.machines.start(machine.id)
                        self._client.machines.wait_until_ready(machine.id)
                except NotFoundError as e:
                    if getattr(e, "elsewhere", None):
                        # Don't silently make a second machine of the same
                        # name in this region: the user meant the other one.
                        raise
                    logger.info("Creating machine %r", self.machine_name)
                    options = {k: v for k, v in self._create_options.items() if v is not None}
                    machine = self._client.machines.create(self.machine_name, **options)
                    self._client.machines.wait_until_ready(machine.id, timeout=180.0)
                self._machine = machine
            return self._machine

    @property
    def machine_id(self) -> str:
        """The managed machine's id (creates the machine if needed)."""
        return self.get_machine().id

    def destroy_machine(self) -> None:
        """Delete the managed machine and all of its state. Irreversible."""
        with self._lock:
            target = self._machine.id if self._machine is not None else self.machine_name
            try:
                self._client.machines.delete(target)
                logger.info("Deleted machine %r", self.machine_name)
            except NotFoundError:
                pass
            self._machine = None

    def fork(self, machine_name: str | None = None, **kwargs: Any) -> BoxdPlugin:
        """Live-fork the managed machine and return a plugin bound to the copy.

        The fork copies the disk, memory and running processes, so the copy
        starts exactly where this machine is now. Give each agent that should
        work on the copy the returned plugin's tools. Keyword arguments are
        passed to :class:`BoxdPlugin` (for example ``destroy_on_close``).

        An unnamed fork gets an ``adk-`` name and is deleted when the
        returned plugin is closed, like an unnamed ``BoxdPlugin()``.
        """
        source = self.get_machine()
        name = machine_name or f"adk-{uuid.uuid4().hex[:8]}"
        labels = self._create_options["labels"]
        delete_after = kwargs.pop("delete_after", self._create_options["delete_after"])
        forked = self._client.machines.fork(
            source.id,
            name,
            labels=labels,
            **({"delete_after": delete_after} if delete_after is not None else {}),
        )
        self._client.machines.wait_until_ready(forked.id, timeout=180.0)
        kwargs.setdefault("plugin_name", self.name)
        if machine_name is None:
            kwargs.setdefault("destroy_on_close", True)
        child = BoxdPlugin(machine_name=name, client=self._client, **kwargs)
        child._machine = forked
        return child

    def save_snapshot(self, name: str, *, wait: bool = True, timeout: float = 600.0) -> str:
        """Save the managed machine's memory and disk as a named snapshot.

        Start new machines from it with ``BoxdPlugin(from_snapshot=name)``.
        Saving the same name again stores a new version. With ``wait`` (the
        default) this returns once the snapshot can be restored. Returns the
        snapshot's name.
        """
        machine = self.get_machine()
        self._client.snapshots.create(machine.id, name)
        if wait:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    status = self._client.snapshots.get(name).status
                except NotFoundError:  # not listed until the capture starts
                    status = "pending"
                if status == "ready":
                    break
                if status == "failed":
                    raise RuntimeError(f"snapshot {name!r} failed")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"snapshot {name!r} not ready after {timeout:g}s")
                time.sleep(2.0)
        return name

    async def close(self) -> None:
        """Release resources; deletes the machine if ``destroy_on_close``."""
        if self._destroy_on_close:
            await asyncio.to_thread(self.destroy_machine)
        if self._owns_client:
            await asyncio.to_thread(self._client.close)

    # -- ADK surface ----------------------------------------------------------

    def get_tools(self) -> list[BaseTool]:
        """Return the boxd tools to pass to an ADK ``Agent``."""
        return list(self._tools)

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        if self._is_ours(tool):
            logger.debug("boxd tool %s args=%s", tool.name, tool_args)
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self._is_ours(tool) and isinstance(result, dict) and result.get("error"):
            logger.warning("boxd tool %s returned error: %s", tool.name, result["error"])
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        if not self._is_ours(tool):
            return None
        logger.error("boxd tool %s raised: %s", tool.name, error)
        # Surface the failure to the model as a structured result so the
        # agent can adapt instead of the run aborting.
        return {"success": False, "error": str(error)}

    def _is_ours(self, tool: BaseTool) -> bool:
        return any(t is tool for t in self._tools)

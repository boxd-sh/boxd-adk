"""Google ADK integration for boxd (https://boxd.sh/?ref=boxd-adk).

Gives ADK agents a full Linux machine to work in: a boxd microVM with a
persistent disk, checkpoints, snapshots and live forks.
"""

from .plugin import BoxdPlugin
from .tools import (
    CreateCheckpointTool,
    ExecuteCodeTool,
    ExecuteCommandTool,
    ListCheckpointsTool,
    ReadFileTool,
    RestoreCheckpointTool,
    WriteFileTool,
)

__version__ = "0.1.0"

__all__ = [
    "BoxdPlugin",
    "CreateCheckpointTool",
    "ExecuteCodeTool",
    "ExecuteCommandTool",
    "ListCheckpointsTool",
    "ReadFileTool",
    "RestoreCheckpointTool",
    "WriteFileTool",
    "__version__",
]

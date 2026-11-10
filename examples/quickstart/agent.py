"""Quickstart: an ADK agent with a boxd machine.

Prerequisites:
    pip install boxd-adk
    export BOXD_API_KEY=...    # boxd console -> API keys
    export GOOGLE_API_KEY=...  # for the Gemini model

Run with the ADK CLI from the repository root:
    adk run examples/quickstart     # terminal chat
    adk web examples                # dev UI, pick "quickstart"
or programmatically via a Runner (see persistent_environment.py).

Any ADK model works: set ADK_MODEL=anthropic/claude-sonnet-4-5 (with
`pip install litellm` and ANTHROPIC_API_KEY) to use Claude instead.
"""

import os

from google.adk.agents import Agent
from google.adk.apps import App

from boxd_adk import BoxdPlugin

# No machine_name: an `adk-` prefixed machine is created on first tool use
# and deleted when the plugin is closed (the runner closes it for you).
plugin = BoxdPlugin()

root_agent = Agent(
    model=os.environ.get("ADK_MODEL", "gemini-flash-latest"),
    name="boxd_agent",
    instruction=(
        "You help users build and test code inside a Linux machine (a boxd "
        "machine). Run commands and code in the machine, not locally. Create "
        "a checkpoint before risky operations such as package upgrades or "
        "migrations."
    ),
    tools=plugin.get_tools(),
)

# Registering the plugin on the App gives it lifecycle callbacks and error
# handling, and lets the runner delete the machine when it shuts down.
app = App(name="quickstart", root_agent=root_agent, plugins=[plugin])

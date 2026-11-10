"""A persistent development environment that survives across sessions.

Because the machine is named, every run of this script attaches to the SAME
Linux machine: packages installed yesterday are still installed today, files
are still there, and the agent picks up where it left off. An idle machine
goes to sleep on its own and wakes on the next command, so a parked
environment costs very little.

    export BOXD_API_KEY=...
    export GOOGLE_API_KEY=...
    python examples/persistent_environment.py "create a fastapi project in ~/app"
    # ...later, a new process, same environment:
    python examples/persistent_environment.py "add a /health endpoint to ~/app"
"""

import asyncio
import os
import sys

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.genai import types

from boxd_adk import BoxdPlugin


async def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "What is in /home/boxd/app? Summarize the project state."

    # Named machine: reused across sessions, never deleted automatically.
    plugin = BoxdPlugin(machine_name="my-dev-environment")

    agent = Agent(
        model=os.environ.get("ADK_MODEL", "gemini-flash-latest"),
        name="dev_env_agent",
        instruction=(
            "You maintain a long-lived development environment inside a boxd "
            "machine. All commands, code, and files run inside the machine. "
            "It persists between sessions, so check existing state before "
            "assuming a fresh machine. Create a checkpoint before risky "
            "changes."
        ),
        tools=plugin.get_tools(),
    )

    app = App(name="dev_env", root_agent=agent, plugins=[plugin])
    async with InMemoryRunner(app=app) as runner:  # closes the plugin on exit
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="demo"
        )
        async for event in runner.run_async(
            user_id="demo",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
        ):
            if event.content and event.content.parts and event.content.parts[0].text:
                print(event.content.parts[0].text)
    # The named machine is kept.


if __name__ == "__main__":
    asyncio.run(main())

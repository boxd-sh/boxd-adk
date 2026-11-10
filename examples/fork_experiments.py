"""Fork: try two approaches in parallel from the same starting point.

One machine is set up once. Then it is live-forked twice (disk, memory and
running processes are copied in about a second), and two agents work on the
copies at the same time, each trying a different approach. Neither can break
the other's machine or the original.

    export BOXD_API_KEY=...
    export GOOGLE_API_KEY=...
    python examples/fork_experiments.py
"""

import asyncio
import os

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.genai import types

from boxd_adk import BoxdPlugin

MODEL = os.environ.get("ADK_MODEL", "gemini-flash-latest")

APPROACHES = {
    "recursive": "Implement fib(n) in /home/boxd/fib.py recursively with memoization.",
    "iterative": "Implement fib(n) in /home/boxd/fib.py with a simple loop.",
}


async def run_agent(plugin: BoxdPlugin, name: str, task: str) -> str:
    agent = Agent(
        model=MODEL,
        name=f"{name}_agent",
        instruction="Work only inside the boxd machine, using your tools.",
        tools=plugin.get_tools(),
    )
    prompt = f"{task} Then run `python3 /home/boxd/bench.py` and reply with its output only."
    answer = ""
    app = App(name=name, root_agent=agent, plugins=[plugin])
    async with InMemoryRunner(app=app) as runner:  # deletes the fork on exit
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="demo"
        )
        async for event in runner.run_async(
            user_id="demo",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                answer = "".join(p.text or "" for p in event.content.parts)
    return answer


async def main() -> None:
    base = BoxdPlugin()  # throwaway machine, deleted at the end

    # Shared setup, done once on the base machine.
    base.client.machines.fs.write(
        base.machine_id,
        "/home/boxd/bench.py",
        "import time\nfrom fib import fib\n"
        "t = time.perf_counter(); v = fib(300)\n"
        "print(f'fib(300) ends in {v % 1000}, took {time.perf_counter() - t:.6f}s')\n",
    )

    # One live fork per approach; each fork is deleted when its runner closes.
    forks = {name: await asyncio.to_thread(base.fork) for name in APPROACHES}
    answers = await asyncio.gather(
        *(run_agent(forks[name], name, task) for name, task in APPROACHES.items())
    )
    for name, answer in zip(APPROACHES, answers, strict=True):
        print(f"{name}: {answer}")

    await base.close()


if __name__ == "__main__":
    asyncio.run(main())

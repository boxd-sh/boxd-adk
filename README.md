# boxd-adk

Google [ADK](https://adk.dev/) integration for [boxd](https://boxd.sh/?ref=boxd-adk): give your agent a full Linux machine with a persistent disk, checkpoints, snapshots and live forks.

A boxd machine is a KVM microVM running Ubuntu 24.04, with sudo, Python, Node.js, Docker and a public HTTPS URL. It boots in about a second and keeps everything (files, packages, databases, running services) until you delete it. Idle machines go to sleep on their own and wake on the next command, so your agent can pick up yesterday's environment today. Before a risky change the agent can **checkpoint** the whole machine, memory included, and roll back in under a second if it goes wrong. From your own code you can also **fork** a running machine to try several approaches in parallel.

## Installation

```sh
pip install boxd-adk
```

You'll need a boxd API key ([sign up](https://boxd.sh/?ref=boxd-adk), then create a key in the console or with `boxd auth keys create <name>`):

```sh
export BOXD_API_KEY=...
```

Inside a boxd machine you can skip the key: the SDK signs in as the machine's owner.

## Quickstart

```python
import asyncio

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner

from boxd_adk import BoxdPlugin

plugin = BoxdPlugin()  # or BoxdPlugin(machine_name="my-project") for a persistent machine

root_agent = Agent(
    model="gemini-flash-latest",
    name="boxd_agent",
    instruction="Run code and commands in the boxd machine, not locally.",
    tools=plugin.get_tools(),
)

# Registering the plugin on the App gives it lifecycle callbacks and error
# handling, and lets the runner delete the machine when it shuts down.
app = App(name="boxd_app", root_agent=root_agent, plugins=[plugin])


async def main():
    async with InMemoryRunner(app=app) as runner:
        await runner.run_debug("Use Python to compute 2**100, and tell me which OS you are on.")


asyncio.run(main())
```

Any ADK model works. For Claude, use `model="anthropic/claude-sonnet-4-5"` with `pip install litellm` and `ANTHROPIC_API_KEY` set.

## Two modes

| | `BoxdPlugin()` | `BoxdPlugin(machine_name="my-project")` |
|---|---|---|
| Machine name | auto-generated `adk-…` | yours |
| Reused across sessions | no | **yes, full state persists** |
| Deleted on `plugin.close()` | yes | no |

The machine is created lazily on first tool use; constructing the agent needs no network. An existing machine that is asleep is woken by the first command, and a stopped one is started.

`plugin.close()` runs when the runner shuts down. If your process can die before that, pass `delete_after=3600` (seconds) and boxd deletes the machine on its own after that time, whatever happens to your process.

## Tools

| Tool | Description |
|---|---|
| `execute_command_in_boxd` | Run a shell command (cwd, timeout supported) |
| `execute_code_in_boxd` | Run a Python / JavaScript / bash snippet |
| `write_file_to_boxd` | Write a text file (parents auto-created) |
| `read_file_from_boxd` | Read a text file |
| `create_boxd_checkpoint` | Checkpoint the entire machine: disk, memory and processes |
| `list_boxd_checkpoints` | List checkpoints, newest first |
| `restore_boxd_checkpoint` | Roll back to a checkpoint (**destructive**, requires `confirm=true`) |

All tools return structured dicts (`success`, `stdout`/`stderr`/`exit_code`, …); failures come back as `{"success": false, "error": ...}` so the agent can adapt instead of crashing the run. A command that runs past its timeout is stopped and reported with `"timed_out": true`.

Commands run as the `boxd` user (with passwordless sudo) in `/home/boxd`. Files are read and written through boxd's filesystem API as that same user, so there is no size limit on writes, and relative paths start in `/home/boxd`. Output and file content returned to the model are capped at 100,000 characters.

### A note on restore

`restore_boxd_checkpoint` rewinds the **whole machine**, including memory and running processes, and permanently discards anything newer than the checkpoint. The tool refuses to run unless `confirm=true` is passed, and its description tells the model to get explicit user confirmation first. `create_boxd_checkpoint` waits until the checkpoint is ready, so a restore right after it works.

## Forks and snapshots

These are Python methods on the plugin rather than agent tools, because each one gives you a *new* machine, and a set of tools is bound to one machine.

```python
base = BoxdPlugin()
# ...set up the project once...

fork = base.fork()            # live copy: disk, memory and running processes
other_agent = Agent(model="gemini-flash-latest", name="other", tools=fork.get_tools())

base.save_snapshot("project-base")                      # reusable starting point
fresh = BoxdPlugin(from_snapshot="project-base")        # new machines start from it
```

An unnamed fork is deleted when its plugin is closed. See [`examples/fork_experiments.py`](examples/fork_experiments.py) for two agents working on two forks in parallel.

## Configuration

```python
BoxdPlugin(
    api_key=None,              # default: $BOXD_API_KEY (not needed inside a boxd machine)
    machine_name=None,         # default: auto-generated "adk-…"
    plugin_name="boxd_plugin",
    region=None,               # default: $BOXD_REGION, then the API's own region
    base_url=None,             # default: $BOXD_BASE_URL, then the public boxd API
    from_snapshot=None,        # new machines: start from this snapshot
    image=None,                # new machines: boot this image instead
    vcpu=None,                 # new machines: 1, 2, 4 or 8 vCPUs (4 GB RAM each)
    labels=None,               # new machines: extra labels
    delete_after=None,         # new machines: seconds until boxd deletes it anyway
    destroy_on_close=None,     # default: True if unnamed, False if named
    client_timeout=60.0,       # timeout for boxd API calls
    client=None,               # an existing boxd.Boxd client to share
)
```

`plugin.client` is the underlying [boxd SDK](https://pypi.org/project/boxd/) client and `plugin.machine_id` the machine's id, for anything the tools don't cover (exposing a port, egress allowlists, pausing).

## Examples

All examples read the model from `ADK_MODEL` (default `gemini-flash-latest`).

- [`examples/quickstart/`](examples/quickstart/agent.py): minimal agent, run with `adk run examples/quickstart` or `adk web examples`
- [`examples/persistent_environment.py`](examples/persistent_environment.py): a named dev environment that survives across sessions
- [`examples/checkpoint_rollback.py`](examples/checkpoint_rollback.py): checkpoint, break things, roll back
- [`examples/fork_experiments.py`](examples/fork_experiments.py): fork one machine and let two agents try different approaches in parallel

## Attribution

Machines created by this plugin carry the label `created-by=google-adk`, so you can find them (`boxd machine list --label created-by=google-adk`) and boxd can see how much usage comes from ADK agents. Pass your own `created-by` label to override it.

## Development

```sh
pip install -e '.[dev]'
pytest                                                # unit tests, no network
BOXD_ADK_LIVE=1 BOXD_API_KEY=... pytest tests/test_live.py -v   # against real boxd
```

The live tests create machines named `t-adk-…` with a 30 minute delete-after timer and delete them at the end. The agent test also needs `ANTHROPIC_API_KEY` and `pip install litellm`.

## Resources

- [boxd docs](https://docs.boxd.sh/?ref=boxd-adk)
- [boxd Python SDK](https://pypi.org/project/boxd/)
- [Google ADK docs](https://adk.dev/)

## License

MIT

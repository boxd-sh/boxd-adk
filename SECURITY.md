# Security

Please don't report security problems in public issues.

Report them privately through
[GitHub's vulnerability reporting](https://github.com/boxd-sh/boxd-adk/security/advisories/new),
or email [contact@boxd.sh](mailto:contact@boxd.sh). Include what you found, how
to reproduce it, and the version of `boxd-adk` you use. We'll confirm we
received it and keep you posted while we fix it.

## Supported versions

Fixes go into the latest release.

## Scope

This repo holds the `boxd-adk` package and its examples. Issues in boxd itself,
such as the API or the machines, can go to the same addresses.

## Running agents safely

The tools let a model run any command on its machine. That machine is a full
VM with its own kernel, so a command can't reach your own computer, but it can
reach the internet and anything the machine can. Give the agent a machine you
are happy for it to change, use checkpoints before risky steps, and pass
`delete_after` so a machine you forget about deletes itself.

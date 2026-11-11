# Changelog

## 0.1.0 (unreleased)

- First release: `BoxdPlugin` gives a Google ADK agent a boxd machine through seven tools:
  `execute_command_in_boxd`, `execute_code_in_boxd` (Python, JavaScript, bash),
  `write_file_to_boxd`, `read_file_from_boxd`, `create_boxd_checkpoint`,
  `list_boxd_checkpoints` and `restore_boxd_checkpoint` (only with `confirm=true`).
- Two modes: an unnamed machine deleted when the runner shuts down, or a named machine that
  persists across sessions and wakes on the next command.
- `fork()` and `save_snapshot()` on the plugin, plus `from_snapshot` and `image` for new machines.
- Machine options: `vcpu`, `labels`, `delete_after`, `region`; machines are labelled
  `created-by=google-adk`.
- Failures come back to the model as `{"success": false, "error": ...}` instead of ending the run.
- Ships `py.typed`.

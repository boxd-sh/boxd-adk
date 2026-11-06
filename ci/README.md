# CI workflows (to move into place)

`test.yml` (lint, unit tests and build on Python 3.10 and 3.13) and
`publish.yml` (publish to PyPI on a `v*` tag, via trusted publishing) belong in
`.github/workflows/`. They live here because the token that pushed this branch
had no `workflow` scope. Move them with:

```sh
git mv ci/test.yml ci/publish.yml .github/workflows/ && git rm ci/README.md
```

# Publishing

Nothing here is published yet.

1. Check everything, against real boxd too:

   ```sh
   pip install -e '.[dev]' build twine litellm
   ruff check . && ruff format --check .
   pytest
   BOXD_ADK_LIVE=1 BOXD_API_KEY=... ANTHROPIC_API_KEY=... pytest tests/test_live.py -v
   ```

   The live tests run against production unless `BOXD_BASE_URL` is set. Then run
   the quickstart from the README and the three examples with a real model
   (`GOOGLE_API_KEY` set).

2. Build and check the package:

   ```sh
   rm -rf dist && python -m build
   twine check dist/*
   unzip -l dist/*.whl
   ```

   The wheel holds `boxd_adk/` with `py.typed`, and the metadata shows the README.

3. Install the wheel in a clean virtualenv and run the README quickstart from
   it, so you test what users get.

4. Set the date in `CHANGELOG.md`. The first release ships `0.1.0` as it stands
   in `pyproject.toml`; bump `version` there for every later release.

5. The banner at the top of the README to its
   full `raw.githubusercontent.com` URL: PyPI doesn't show relative images.

6. Publish, then tag:

   ```sh
   twine upload dist/*
   git tag v0.1.0 && git push --tags
   ```

7. Mark the ADK docs pull request (google/adk-docs#2572) ready for review.

pip install uv
uv venv
source .venv/bin/activate
uv sync
uv pip install -e .
uv cache prune
uv run tests/test_env.py
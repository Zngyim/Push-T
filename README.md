# Push-T Imitation Learning

## 成果展示

### Behavior Cloning

![demo_bc](demo_bc.gif)

### Flow Matching

![demo_fm](demo_fm.gif)



## Setup

This project uses `uv` for package management. `uv` is a Python package and environment manager from [Astral](https://astral.sh). It replaces tools like
`pip`, `pipx`, `conda`, and `virtualenv` with a single, simple interface. It is also much faster than prior tools.

### Installing `uv`

Run the following in your terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, open a new terminal so `uv` is on your `PATH`.

### Always use `uv run`

Do **not** run `python` or `pip` directly. Always run scripts through `uv run` so dependencies
and environments are handled automatically. If you want to add a new dependency, you can use `uv add`. This will add the dependency to `pyproject.toml`, update `uv.lock`, and install the package into your virtual environment.

Example:

```bash
uv run src/push_t_imitation/train.py --help
```

This should work out of the box with the current codebase.

## Weights & Biases (wandb) login

This project uses [Weights & Biases (WandB)](https://wandb.ai) for experiment tracking. WandB is a tool for logging and visualizing machine learning experiments. It is free for academic use. Before running a training script, you will need to log in to WandB using your API key.

```bash
uv run wandb login
```

Follow the prompt to paste your API key.

## Using Modal

**Modal is optional for this project. In testing, training was often faster on a local laptop CPU than on Modal, but you can still use Modal if you want a remote training workflow:**

First, create a Modal account. Then, you can launch remote training with the following command:

```bash
uv run modal run src/push_t_imitation/modal_train.py
```

This will build a Modal container and launch training remotely. You can pass the same flags as the local training script. If you are logged into WandB locally, your API key will be automatically forwarded to the Modal container.

Logs and checkpoints will be saved to a Modal volume called `push_t_imitation_volume`. To inspect the logs, you can use:

```bash
uv run modal volume ls push_t_imitation_volume exp
```

Then, you can download the logs and checkpoints to your local machine using a command like the following:

```bash
uv run modal volume get push_t_imitation_volume exp/<experiment_name>
```
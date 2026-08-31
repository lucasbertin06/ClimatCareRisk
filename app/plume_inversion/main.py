"""Command-line entrypoint for the plume inversion pipeline."""

import typer

from plume_inversion.optimize import recover
from plume_inversion.scenario import make_scenario

app = typer.Typer(name="plume_inversion", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Differentiable plume-source inversion tools."""


@app.command()
def run(steps: int = typer.Option(120, min=1, max=500)) -> None:
    """Run a small differentiable source recovery experiment."""
    scenario = make_scenario()
    result = recover(scenario, steps=steps, learning_rate=0.02)
    typer.echo(f"initial_loss={float(result['losses'][0]):.6f}")
    typer.echo(f"final_loss={float(result['losses'][-1]):.6f}")


def entrypoint() -> None:
    """CLI entrypoint for the application."""
    app()


if __name__ == "__main__":
    entrypoint()

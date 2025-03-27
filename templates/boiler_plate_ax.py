"""
This is boiler plate code for running an experiment with Ax, in which we would
like to fine-tune hyperparameters of a model. This code is meant to be a starting point.

Author: Zach Calhoun
Date: 2025-03-27

Main spots where you will need to modify the code are marked with TODOs
"""

import os
import logging
import argparse
from datetime import datetime

from ax.service.ax_client import AxClient, ObjectiveProperties


def main(args):
    """Run the main experiment loop."""

    client = set_up_experiment()

    # Set up logging -- this is optional, and depends on the extent to which you would
    # like to see the code output as it runs.
    logging.basicConfig(level=logging.INFO)

    # Define the number of trials to run -- you could also define an alternative
    # stopping criteria instead of looping over the number of trials.
    for i in range(args.num_trials):
        objective_value = run_trial(client, i)
        logging.info(
            "Trial %d/%d completed with objective value %s.",
            i + 1,
            args.num_trials,
            objective_value,
        )

    complete_experiment(client, args)


def complete_experiment(client, args):
    """
    Complete the experiment and save the results.

    Optionally do something else at the end of the experiment.
    """
    best_parameters, values = client.get_best_parameters()

    logging.info("Best parameters: %s", best_parameters)
    logging.info("Best values: %s", values)

    # Save the experimental values to a file
    filename = f"ax_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    file_path = os.path.join(args.output_dir, filename)
    client.save_to_json_file(file_path)


def run_trial(client, trial_index):
    """
    Basic training loop:
    1. Initialize trial.
    2. Train and evaluate the model.
    3. Complete the trial.

    Optionally -- you could set this up to run in parallel.
    """
    parameters, trial_index = client.get_next_trial()

    # Train and evaluate the model
    objective_value = train_evaluate(parameters)

    # Complete the trial
    client.complete_trial(trial_index=trial_index, raw_data=objective_value)

    return objective_value


def train_evaluate(parameters):
    """
    This function trains and evaluates the model, and returns the objective value
    that we would like to optimize.

    TODO: Set up your model to be trained and evaluated.
    """

    # Fit the model
    # E.g., fit a CNN model.

    # Evaluate the model

    # Return the objective function as evaluated on the model
    # Example:
    # return {"f": f(x)}

    return NotImplementedError("You must implement the train_evaluate function.")


def set_up_experiment():
    """
    This function sets up the experiment -- modify to define the parameters that
    you care about, and the objective function that you would like to optimize.

    Check out https://ax.dev/docs/tutorials/tune_cnn_service/ for a tutorial.

    TODO: Define the parameters and objective function that you would like to optimize.
    """
    ax_client = AxClient()
    ax_client.create_experiment(
        name="name_your_experiment",
        parameters=[
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [1e-4, 1e-1],
                "value_type": "float",
                "log_scale": True,
            },
            {
                "name": "dropout",
                "type": "range",
                "bounds": [0.1, 0.5],
                "value_type": "float",
            },
            {
                "name": "batch_size",
                "type": "choice",
                "values": [32, 64, 128],
                "value_type": "int",
            },
        ],
        objectives={"mse": ObjectiveProperties(minimize=True)},
    )
    return ax_client


def parse_args():
    """This is a common way to pass arguments to a script."""
    parser = argparse.ArgumentParser(description="Run an experiment with Ax.")
    parser.add_argument(
        "--num_trials", type=int, default=10, help="Number of trials to run."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the results of the experiment",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed_arguments = parse_args()
    main(parsed_arguments)

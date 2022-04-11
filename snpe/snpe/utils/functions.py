import copy
import subprocess

import torch


# A utility function to execute a command on the terminal using subprocess
# and print the outputs from the terminal in jupyter/python
# Use it as:
# for path in terminal_execute(command):
#   print(path, end="")
# https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
def terminal_execute(cmd: str) -> None:
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


# A utility function to check the convergence of a torch model tranining process. If the validation loss does not decrease
# for a number of epochs, training is stopped
# The torch model does need to have a best_validation_loss and epochs since last improvement
# defined in __init__ for this to work
def nn_converged(epoch: int, stop_after_epochs: int, validation_loss: torch.Tensor, model: torch.nn.Module):
    converged = False
    # (Re)-start the epoch count with the first epoch or any improvement.
    if epoch == 0 or validation_loss < model.best_validation_loss:
        model.best_validation_loss = validation_loss
        model.epochs_since_last_improvement = 0
        model.best_model = copy.deepcopy(model.net.state_dict())
    else:
        model.epochs_since_last_improvement += 1

    # If no validation improvement over many epochs, stop training.
    if model.epochs_since_last_improvement > stop_after_epochs - 1:
        model.net.load_state_dict(model.best_model)
        converged = True
    return converged

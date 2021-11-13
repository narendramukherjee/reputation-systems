import subprocess


# A utility function to execute a command on the terminal using subprocess
# and print the outputs from the terminal in jupyter/python
# Use it as:
# for path in terminal_execute(command):
#   print(path, end="")
# https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
def terminal_execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

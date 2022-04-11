import contextlib

from multiprocessing import Queue
from typing import Any, List

import joblib

from tqdm.auto import tqdm


# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
# tqdm custom context for joblib jobs
@contextlib.contextmanager
def tqdm_joblib(tqdm_object) -> Any:
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs) -> Any:
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# Custom multi-progressbar version of tqdm to track progress of long running processes
# Modified from the batch processing example in:
# https://towardsdatascience.com/parallel-batch-processing-in-python-8dcce607d226
# Define the multi-progressbar that will then be directly used in the marketplace simulations (For eg.)
def multi_progressbar(
    totals: List,
    queue: Queue,
) -> None:
    pbars = [
            tqdm(
                desc=f'Worker {pid + 1}',
                total=total,
                position=pid,
            )
            for pid, total in enumerate(totals)
        ]

    while True:
        try:
            message = queue.get()
            if message.startswith('update'):
                pid = int(message[6:])
                pbars[pid].update(1)
            elif message == 'done':
                break
        except:
            pass
    for pbar in pbars:
        pbar.close()

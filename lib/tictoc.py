import time


class TicToc:
    def __init__(self):
        self._start_time = time.perf_counter()

    @property
    def start_time(self):
        """Returns the start time of the timer."""
        return self._start_time

    def toc(self, print_elapsed=True):
        if self._start_time is None:
            raise RuntimeError("tic() must be called before toc()")
        elapsed = time.perf_counter() - self._start_time
        if print_elapsed:
            print(f"Elapsed time: {elapsed:.6f} seconds")
        return elapsed

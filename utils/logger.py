from pathlib import Path
from time import perf_counter as timer
from datetime import datetime
from collections import OrderedDict
import numpy as np


class Logger:
    """
    Logger for classifier
    """
    def __init__(self, root):
        self.text_file = open(Path(root, "log.txt"), "w")

        start_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line(f"Creating log on {start_time}")
        self.write_line("=".center(100, '='))

    def write_line(self, line):
        self.text_file.write("%s\n" % line)

    def finalize(self):
        self.write_line("=".center(100, '='))
        end_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line(f"Finished on {end_time}")
        self.text_file.close()


class Profiler:
    def __init__(self, summarize_every=5, disabled=False):
        self.last_tick = timer()
        self.logs = OrderedDict()
        self.summarize_every = summarize_every
        self.disabled = disabled

    def tick(self, name):
        if self.disabled:
            return

        # Log the time needed to execute that function
        if not name in self.logs:
            self.logs[name] = []
        if len(self.logs[name]) >= self.summarize_every:
            self.summarize()
            self.purge_logs()
        self.logs[name].append(timer() - self.last_tick)

        self.reset_timer()

    def purge_logs(self):
        for name in self.logs:
            self.logs[name].clear()

    def reset_timer(self):
        self.last_tick = timer()

    def summarize(self):
        n = max(map(len, self.logs.values()))
        assert n == self.summarize_every
        print("\nAverage execution time over %d steps:" % n)

        name_msgs = [
            "%s (%d/%d):" % (name, len(deltas), n)
            for name, deltas in self.logs.items()
        ]
        pad = max(map(len, name_msgs))
        for name_msg, deltas in zip(name_msgs, self.logs.values()):
            print("  %s  mean: %4.0fms   std: %4.0fms" %
                  (name_msg.ljust(pad), np.mean(deltas) * 1000,
                   np.std(deltas) * 1000))
        print("", flush=True)

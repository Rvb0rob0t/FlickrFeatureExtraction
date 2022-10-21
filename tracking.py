import logging
import time
import threading
import tensorflow as tf


class Tracker:

    def __init__(self, mins_to_update=1):
        self.logger = logging.getLogger(__name__)
        self.speed_counter_names = set()
        self.last_counters = {}
        self.counters = {}
        self.mins_to_update = mins_to_update
        self.restart()

    def restart(self):
        self.start_time = time.time()
        self.last_update = 0
        self.reset_counts()

    def reset_counts(self):
        for counter_name in self.counters:
            self.counters[counter_name] = 0

    def register_speeds(self, *counter_names):
        self.speed_counter_names.update(counter_names)
        for counter_name in counter_names:
            if counter_name not in self.counters:
                self.counters[counter_name] = 0

    def increment(self, counter_name, amount=1):
        if counter_name not in self.counters:
            self.counters[counter_name] = amount
        else:
            self.counters[counter_name] += amount

    def get_counter(self, counter_name):
        return self.counters.get(counter_name)

    def get_counters(self):
        return self.counters.copy()

    def track_progress(self):
        elapsed_mins = int((time.time() - self.start_time) / 60)
        mins_since_last_update = elapsed_mins - self.last_update
        if mins_since_last_update >= self.mins_to_update:
            speeds = {}
            for k in self.speed_counter_names:
                if k in self.last_counters:
                    d = self.counters[k] - self.last_counters[k]
                    speeds[k] = d / mins_since_last_update
                else:
                    speeds[k] = self.counters[k]
            s = f"Update for min {elapsed_mins}: "
            for k, v in self.counters.items():
                s += f"{k}={v:,}; "
                if k in speeds:
                    s += f"{k}_per_min={speeds[k]:,}; "
            self.logger.info(s)
            self.last_update = elapsed_mins
            self.last_counters = self.counters.copy()


class ThreadSafeDecorator():

    def __init__(self, tracker: Tracker):
        self.lock = threading.Lock()
        self.tracker = tracker
        self._start_tracker_worker()

    def _start_tracker_worker(self):
        def function():
            while True:
                self._safe_tracker_call(Tracker.track_progress)
                time.sleep(self.tracker.mins_to_update*60)
        w = threading.Thread(target=function, name='Tracker', daemon=True)
        w.start()

    def _safe_tracker_call(self, func, *args, **kwargs):
        # self.logger.debug(
        #     f"Thread {threading.current_thread().name} acquiring Tracker lock...")
        self.lock.acquire()
        # self.logger.debug(
        #     f"Tracker lock acquired by thread {threading.current_thread().name}")
        func(self.tracker, *args, **kwargs)
        # self.logger.debug(
        #     f"Tracker lock released by thread {threading.current_thread().name}")
        self.lock.release()

    def restart(self):
        self._safe_tracker_call(Tracker.restart)

    def reset_counts(self):
        self._safe_tracker_call(Tracker.reset_counts)

    def register_speeds(self, *counter_names):
        self._safe_tracker_call(Tracker.register_speeds, *counter_names)

    def increment(self, counter_name, amount=1):
        self._safe_tracker_call(Tracker.increment, counter_name, amount)

    def track_progress(self):
        pass

    def __getattr__(self, attr):
        """All non-adapted calls are passed to the object"""
        return getattr(self.tracker, attr)

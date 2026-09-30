from __future__ import annotations

import multiprocessing
import threading
from collections.abc import Callable, Sequence

_PROCESS_SHARED_COUNTER = None


def initialize_process_counter(shared) -> None:
    global _PROCESS_SHARED_COUNTER
    _PROCESS_SHARED_COUNTER = shared


def record_work(units: int, counter=None) -> None:
    if counter is not None:
        counter.add(units)
    elif _PROCESS_SHARED_COUNTER is not None:
        WorkCounter.add_shared(_PROCESS_SHARED_COUNTER, units)


class ProgressReporter:
    """Map normalized progress updates to a monotonic public percentage."""

    def __init__(
        self,
        callback: Callable[[float], None] | None,
        start: float = 0.0,
        end: float = 100.0,
        min_step: float = 0.5,
        _root=None,
    ):
        if start > end:
            raise ValueError("start must be less than or equal to end")
        if min_step < 0:
            raise ValueError("min_step must be non-negative")

        self.callback = callback if _root is None else _root.callback
        self.start = float(start)
        self.end = float(end)
        self.min_step = float(min_step)
        self._root = self if _root is None else _root
        if _root is None:
            self._lock = threading.RLock()
            self._last_emitted = None

    def update(self, fraction: float) -> None:
        fraction = max(0.0, min(1.0, float(fraction)))
        value = self.start + (self.end - self.start) * fraction
        self._root._emit(value, force=fraction == 1.0)

    def _emit(self, value: float, force: bool = False) -> None:
        if self.callback is None:
            return

        with self._lock:
            if self._last_emitted is not None:
                if value <= self._last_emitted:
                    return
                if not force and value - self._last_emitted < self.min_step:
                    return
            self._last_emitted = value
            self.callback(value)

    def child(self, start_frac: float, end_frac: float) -> ProgressReporter:
        start_frac = max(0.0, min(1.0, float(start_frac)))
        end_frac = max(0.0, min(1.0, float(end_frac)))
        if start_frac > end_frac:
            raise ValueError("start_frac must be less than or equal to end_frac")
        width = self.end - self.start
        return ProgressReporter(
            self.callback,
            self.start + width * start_frac,
            self.start + width * end_frac,
            self.min_step,
            _root=self._root,
        )

    def split(self, weights: Sequence[float]) -> list[ProgressReporter]:
        if len(weights) == 0:
            return []
        normalized_weights = [float(weight) for weight in weights]
        if any(weight < 0 for weight in normalized_weights):
            raise ValueError("weights must be non-negative")
        total_weight = sum(normalized_weights)
        if total_weight <= 0:
            raise ValueError("weights must have a positive sum")

        children = []
        start_frac = 0.0
        for weight in normalized_weights:
            end_frac = start_frac + weight / total_weight
            children.append(self.child(start_frac, end_frac))
            start_frac = end_frac
        return children

    def finish(self) -> None:
        self.update(1.0)


class WorkCounter:
    """Count completed pixel units and report them from the coordinator."""

    def __init__(self, total_units: int, reporter: ProgressReporter, shared=None):
        if total_units < 0:
            raise ValueError("total_units must be non-negative")
        self.total_units = int(total_units)
        self.reporter = reporter
        self.shared = shared
        self._done = 0
        self._lock = threading.Lock()

    @staticmethod
    def make_shared(context=None):
        context = context or multiprocessing.get_context("spawn")
        return context.Value("q", 0)

    @staticmethod
    def add_shared(shared, units: int) -> None:
        if units < 0:
            raise ValueError("units must be non-negative")
        with shared.get_lock():
            shared.value += units

    @property
    def done(self) -> int:
        if self.shared is not None:
            with self.shared.get_lock():
                return int(self.shared.value)
        with self._lock:
            return self._done

    def add(self, units: int) -> None:
        if units < 0:
            raise ValueError("units must be non-negative")
        if self.shared is not None:
            self.add_shared(self.shared, units)
        else:
            with self._lock:
                self._done += units

    def poll(self) -> None:
        if self.total_units == 0:
            self.reporter.update(1.0)
            return
        self.reporter.update(self.done / self.total_units)

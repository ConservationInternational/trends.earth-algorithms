import multiprocessing
import threading

from te_algorithms.gdal.progress import ProgressReporter, WorkCounter

_PROCESS_COUNTER = None


def _set_process_counter(counter):
    global _PROCESS_COUNTER
    _PROCESS_COUNTER = counter


def _increment_process_counter(units):
    WorkCounter.add_shared(_PROCESS_COUNTER, units)


def test_reporter_is_monotonic_and_clamps_values():
    values = []
    reporter = ProgressReporter(values.append, min_step=0)

    reporter.update(0.4)
    reporter.update(0.2)
    reporter.update(5)
    reporter.update(-5)

    assert values == [40.0, 100.0]


def test_reporter_child_and_split_ranges():
    values = []
    reporter = ProgressReporter(values.append, min_step=0)
    first, second = reporter.split([1, 3])

    first.finish()
    second.update(0.5)
    second.finish()

    assert values == [25.0, 62.5, 100.0]


def test_reporter_throttles_small_updates_but_finishes():
    values = []
    reporter = ProgressReporter(values.append, min_step=1)

    reporter.update(0.005)
    reporter.update(0.01)
    reporter.finish()

    assert values == [0.5, 100.0]


def test_reporter_serializes_concurrent_updates():
    values = []
    reporter = ProgressReporter(values.append, min_step=0)
    threads = [
        threading.Thread(target=reporter.update, args=(fraction / 100,))
        for fraction in range(1, 101)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert values == sorted(values)
    assert values[-1] == 100.0


def test_reporter_accepts_no_callback():
    ProgressReporter(None).update(0.5)
    ProgressReporter(None).finish()


def test_work_counter_thread_updates():
    values = []
    counter = WorkCounter(10, ProgressReporter(values.append, min_step=0))

    counter.add(3)
    counter.add(2)
    counter.poll()

    assert counter.done == 5
    assert values == [50.0]


def test_work_counter_process_shared_value():
    context = multiprocessing.get_context("spawn")
    shared = WorkCounter.make_shared(context)
    counter = WorkCounter(10, ProgressReporter(None), shared=shared)

    with context.Pool(2, initializer=_set_process_counter, initargs=(shared,)) as pool:
        pool.map(_increment_process_counter, [1, 2, 3, 4])

    assert counter.done == 10

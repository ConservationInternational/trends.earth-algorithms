import multiprocessing
import threading
from pathlib import Path
from types import SimpleNamespace

from te_algorithms.gdal.land_deg import land_deg_progress
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


def test_status_summary_thread_dispatch_preserves_region_index(monkeypatch):
    class DatelineAoi:
        def meridian_split(self, **_kwargs):
            return ["west", "east"]

        def get_aligned_output_bounds(self, _compute_bbs_from):
            return [(0, 0, 1, 1), (1, 0, 2, 1)]

    status = {
        "sdg_summaries": [{}],
        "prod_summaries": [{"all_cover_types": {}, "non_water": {}}],
        "lc_summaries": [{}],
        "soc_summaries": [{"all_cover_types": {}, "non_water": {}}],
    }
    change = {
        "sdg_crosstabs": [{}],
        "prod_crosstabs": [{}],
        "lc_crosstabs": [{}],
        "soc_crosstabs": [{}],
    }
    periods = [
        {
            "params": {
                "periods": {
                    indicator: {"year_initial": 2000, "year_final": 2005}
                    for indicator in ("productivity", "land_cover", "soc")
                }
            }
        },
        {
            "params": {
                "periods": {
                    indicator: {"year_initial": 2006, "year_final": 2010}
                    for indicator in ("productivity", "land_cover", "soc")
                }
            }
        },
    ]
    fake_dataset = SimpleNamespace(GetGeoTransform=lambda: (0, 1, 0, 0, 0, -1))
    monkeypatch.setattr(
        land_deg_progress,
        "_get_status_summary_input_vrt",
        lambda *_args: (Path("status.vrt"), {}),
    )
    monkeypatch.setattr(land_deg_progress.gdal, "Open", lambda *_args: fake_dataset)
    monkeypatch.setattr(
        land_deg_progress.gdal, "BuildVRT", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        land_deg_progress,
        "_process_single_region",
        lambda region: (status, change, f"reporting-{region[0]}.tif"),
    )

    result = land_deg_progress.compute_status_summary(
        df=None,
        prod_mode=None,
        job_output_path=Path("summary.tif"),
        aoi=DatelineAoi(),
        compute_bbs_from=None,
        periods=periods,
        nesting=None,
        n_cpus=2,
        parallel_backend="thread",
    )

    assert result[2].path == Path("summary_reporting.vrt")

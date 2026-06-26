"""
Tests for te_algorithms.gee.download._download_default.

Covers the three logical branches:
  1. band_number provided  – selects a specific band, uses band_name or name
  2. no band_number, single-band asset – uses band_name/band_metadata/band_add_to_map
  3. no band_number, multi-band asset  – falls back to name for every band (original
     behaviour preserved so existing datasets are unaffected)
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy dependencies before any te_algorithms import
# ---------------------------------------------------------------------------
if "ee" not in sys.modules:
    sys.modules["ee"] = MagicMock()
if "te_schemas" not in sys.modules:
    sys.modules["te_schemas"] = MagicMock()
    sys.modules["te_schemas.results"] = MagicMock()
    sys.modules["te_schemas.schemas"] = MagicMock()

from te_algorithms.gee.download import _download_default  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image_mock(n_bands: int, properties: dict = None):
    """Return a mock ee.Image whose getInfo() describes n_bands bands."""
    img = MagicMock()
    img.getInfo.return_value = {
        "properties": properties or {"source": "test"},
        "bands": [{"id": f"b{i + 1}"} for i in range(n_bands)],
    }
    img.select.return_value = img  # select returns same mock
    return img


def _run(n_bands=1, properties=None, **kwargs):
    """
    Call _download_default with a controlled ee.Image mock and return the
    (band_info_calls, te_image_calls) captured during the call.

    band_info_calls  – list of call objects for every BandInfo(...) invocation
    te_image_calls   – call args for the TEImage(...) constructor
    """
    img_mock = _make_image_mock(n_bands, properties)

    with (
        patch("te_algorithms.gee.download.ee") as mock_ee,
        patch("te_algorithms.gee.download.BandInfo") as mock_band_info,
        patch("te_algorithms.gee.download.TEImage") as mock_te_image,
        patch("te_algorithms.gee.download.teimage_v1_to_teimage_v2") as mock_conv,
    ):
        mock_ee.Image.return_value = img_mock
        mock_conv.side_effect = lambda x: x  # pass-through

        _download_default(
            asset="test/asset",
            name="Dataset Title",
            temporal_resolution="one-time",
            year_initial=2008,
            year_final=2023,
            **kwargs,
        )

        return mock_band_info.call_args_list, mock_te_image.call_args_list


# ---------------------------------------------------------------------------
# band_number branch
# ---------------------------------------------------------------------------


class TestDownloadDefaultWithBandNumber:
    def test_uses_band_name_when_provided(self):
        calls, _ = _run(n_bands=3, band_number=2, band_name="My Band Name")
        assert len(calls) == 1
        assert calls[0].args[0] == "My Band Name"

    def test_falls_back_to_name_when_band_name_absent(self):
        calls, _ = _run(n_bands=3, band_number=2)
        assert calls[0].args[0] == "Dataset Title"

    def test_band_metadata_merged_into_metadata(self):
        extra = {"year_initial": 2008, "year_final": 2023}
        calls, _ = _run(
            n_bands=1,
            properties={"source": "test"},
            band_number=1,
            band_metadata=extra,
        )
        metadata = calls[0].kwargs["metadata"]
        assert metadata["year_initial"] == 2008
        assert metadata["year_final"] == 2023
        assert metadata["source"] == "test"  # image properties preserved

    def test_band_number_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            _run(n_bands=2, band_number=5)

    def test_band_number_zero_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            _run(n_bands=2, band_number=0)


# ---------------------------------------------------------------------------
# single-band, no band_number branch  (the FWv2 / JRC standalone case)
# ---------------------------------------------------------------------------


class TestDownloadDefaultSingleBandNoBandNumber:
    def test_uses_band_name_when_provided(self):
        """band_name should override the dataset title as the band name."""
        calls, _ = _run(n_bands=1, band_name="Land Productivity Dynamics FWv2")
        assert len(calls) == 1
        assert calls[0].args[0] == "Land Productivity Dynamics FWv2"

    def test_falls_back_to_name_when_band_name_absent(self):
        calls, _ = _run(n_bands=1)
        assert calls[0].args[0] == "Dataset Title"

    def test_band_metadata_merged_when_provided(self):
        extra = {"year_initial": 2008, "year_final": 2023}
        calls, _ = _run(
            n_bands=1,
            properties={"source": "test"},
            band_metadata=extra,
        )
        metadata = calls[0].kwargs["metadata"]
        assert metadata["year_initial"] == 2008
        assert metadata["year_final"] == 2023
        assert metadata["source"] == "test"

    def test_band_metadata_not_provided_leaves_image_properties(self):
        calls, _ = _run(n_bands=1, properties={"source": "test"})
        metadata = calls[0].kwargs["metadata"]
        assert metadata == {"source": "test"}

    def test_add_to_map_false_when_requested(self):
        calls, _ = _run(n_bands=1, band_add_to_map=False)
        assert calls[0].kwargs["add_to_map"] is False

    def test_add_to_map_defaults_to_true(self):
        calls, _ = _run(n_bands=1)
        assert calls[0].kwargs["add_to_map"] is True

    def test_add_to_map_true_explicit(self):
        calls, _ = _run(n_bands=1, band_add_to_map=True)
        assert calls[0].kwargs["add_to_map"] is True


# ---------------------------------------------------------------------------
# multi-band, no band_number branch  (legacy behaviour must be unchanged)
# ---------------------------------------------------------------------------


class TestDownloadDefaultMultiBandNoBandNumber:
    def test_all_bands_named_with_dataset_title(self):
        """band_name must NOT be applied when there are multiple bands and no
        band_number – we cannot know which band the name belongs to."""
        calls, _ = _run(
            n_bands=3, band_name="Should Not Appear", properties={"source": "test"}
        )
        # Every BandInfo constructor call must use the dataset title, never band_name
        assert len(calls) >= 1
        for c in calls:
            assert c.args[0] == "Dataset Title"
            assert c.args[0] != "Should Not Appear"

    def test_single_band_name_not_used_for_multiband(self):
        """Providing band_name with a multi-band asset must not affect naming."""
        calls_multi, _ = _run(n_bands=4, band_name="Specific Band")
        calls_single, _ = _run(n_bands=1, band_name="Specific Band")
        # Single-band: band_name used
        assert calls_single[0].args[0] == "Specific Band"
        # Multi-band: dataset title used instead
        for c in calls_multi:
            assert c.args[0] == "Dataset Title"

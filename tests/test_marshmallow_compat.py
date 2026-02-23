"""Marshmallow compatibility regression tests for te_algorithms.

These tests verify that marshmallow-dataclass-based models defined in
te_algorithms correctly serialize and deserialize.  They serve as a
safety net for marshmallow version upgrades.
"""

import pytest


from te_schemas import SchemaBase


# ===================================================================
# 1. ImageInfo round-trip (te_algorithms.gdal.util)
# ===================================================================


class TestImageInfo:
    """Tests for the ImageInfo dataclass in gdal/util.py."""

    def test_roundtrip(self):
        pytest.importorskip("osgeo", reason="GDAL not installed")
        from te_algorithms.gdal.util import ImageInfo

        obj = ImageInfo(
            x_size=1024,
            y_size=2048,
            x_block_size=256,
            y_block_size=256,
            pixel_width=0.00027,
            pixel_height=-0.00027,
        )
        data = ImageInfo.Schema().dump(obj)
        loaded = ImageInfo.Schema().load(data)
        assert loaded.x_size == 1024
        assert loaded.y_size == 2048
        assert loaded.x_block_size == 256
        assert loaded.pixel_width == pytest.approx(0.00027)
        assert loaded.pixel_height == pytest.approx(-0.00027)

    def test_get_n_blocks(self):
        pytest.importorskip("osgeo", reason="GDAL not installed")
        from te_algorithms.gdal.util import ImageInfo

        obj = ImageInfo(
            x_size=512,
            y_size=512,
            x_block_size=256,
            y_block_size=256,
            pixel_width=1.0,
            pixel_height=-1.0,
        )
        assert obj.get_n_blocks() == 4

    def test_schema_is_callable(self):
        pytest.importorskip("osgeo", reason="GDAL not installed")
        from te_algorithms.gdal.util import ImageInfo

        schema = ImageInfo.Schema()
        assert hasattr(schema, "load")
        assert hasattr(schema, "dump")


# ===================================================================
# 2. SummaryTableLD round-trip (te_algorithms.gdal.land_deg.models)
# ===================================================================


class TestSummaryTableLD:
    """Tests for SummaryTableLD and related models."""

    def _make_summary(self):
        from te_algorithms.gdal.land_deg.models import SummaryTableLD

        return SummaryTableLD(
            soc_by_lc_annual_totals=[{1: 100.0, 2: 200.0}],
            lc_annual_totals=[{1: 50.0}],
            lc_trans_zonal_areas=[{1: 30.0}],
            lc_trans_zonal_areas_periods=[{"baseline": 10.0}],
            lc_trans_prod_bizonal={(1, 2): 5.0},
            sdg_zonal_population_total={1: 1000.0},
            sdg_zonal_population_male={1: 500.0},
            sdg_zonal_population_female={1: 500.0},
            sdg_summary={1: 0.5},
            prod_summary={"baseline": {1: 0.3}},
            lc_summary={1: 0.2},
            soc_summary={"baseline": {1: 0.1}},
        )

    def test_roundtrip(self):
        from te_algorithms.gdal.land_deg.models import SummaryTableLD

        obj = self._make_summary()
        data = SummaryTableLD.Schema().dump(obj)
        loaded = SummaryTableLD.Schema().load(data)

        assert isinstance(loaded, SummaryTableLD)
        assert loaded.sdg_summary == {1: 0.5} or loaded.sdg_summary == {"1": 0.5}

    def test_is_schema_base(self):
        from te_algorithms.gdal.land_deg.models import SummaryTableLD

        assert issubclass(SummaryTableLD, SchemaBase)

    def test_dump_method(self):
        obj = self._make_summary()
        data = obj.dump()
        assert isinstance(data, dict)
        assert "sdg_summary" in data


class TestSummaryTableLDStatus:
    def test_roundtrip(self):
        from te_algorithms.gdal.land_deg.models import SummaryTableLDStatus

        obj = SummaryTableLDStatus(
            sdg_summaries=[{1: 0.5}],
            prod_summaries=[{"baseline": {1: 0.3}}],
            lc_summaries=[{1: 0.2}],
            soc_summaries=[{"baseline": {1: 0.1}}],
        )
        data = SummaryTableLDStatus.Schema().dump(obj)
        loaded = SummaryTableLDStatus.Schema().load(data)
        assert isinstance(loaded, SummaryTableLDStatus)


class TestSummaryTableLDChange:
    def test_roundtrip(self):
        from te_algorithms.gdal.land_deg.models import SummaryTableLDChange

        obj = SummaryTableLDChange(
            sdg_crosstabs=[{(1, 2): 0.5}],
            prod_crosstabs=[{(1, 2): 0.3}],
            lc_crosstabs=[{(1, 2): 0.2}],
            soc_crosstabs=[{(1, 2): 0.1}],
        )
        data = SummaryTableLDChange.Schema().dump(obj)
        loaded = SummaryTableLDChange.Schema().load(data)
        assert isinstance(loaded, SummaryTableLDChange)


class TestSummaryTableLDErrorRecode:
    def test_roundtrip(self):
        from te_algorithms.gdal.land_deg.models import SummaryTableLDErrorRecode

        obj = SummaryTableLDErrorRecode(
            baseline_summary={1: 0.5, -1: 0.3, 0: 0.2},
            report_1_summary={1: 0.4},
            report_2_summary=None,
        )
        data = SummaryTableLDErrorRecode.Schema().dump(obj)
        loaded = SummaryTableLDErrorRecode.Schema().load(data)
        assert isinstance(loaded, SummaryTableLDErrorRecode)
        assert loaded.report_2_summary is None


# ===================================================================
# 3. SummaryTableDrought round-trip (te_algorithms.gdal.drought)
# ===================================================================


class TestSummaryTableDrought:
    def test_roundtrip(self):
        pytest.importorskip("openpyxl", reason="openpyxl not installed")
        pytest.importorskip("osgeo", reason="GDAL not installed")
        from te_algorithms.gdal.drought import SummaryTableDrought

        obj = SummaryTableDrought(
            annual_area_by_drought_class=[{-2: 100.0, -1: 200.0, 0: 300.0}],
            annual_population_by_drought_class_total=[{-2: 1000.0}],
            annual_population_by_drought_class_male=[{-2: 500.0}],
            annual_population_by_drought_class_female=[{-2: 500.0}],
            dvi_value_sum_and_count=(15.0, 3),
        )
        data = SummaryTableDrought.Schema().dump(obj)
        loaded = SummaryTableDrought.Schema().load(data)
        assert isinstance(loaded, SummaryTableDrought)

    def test_is_schema_base(self):
        pytest.importorskip("openpyxl", reason="openpyxl not installed")
        pytest.importorskip("osgeo", reason="GDAL not installed")
        from te_algorithms.gdal.drought import SummaryTableDrought

        assert issubclass(SummaryTableDrought, SchemaBase)

    def test_dump_method(self):
        pytest.importorskip("openpyxl", reason="openpyxl not installed")
        pytest.importorskip("osgeo", reason="GDAL not installed")
        from te_algorithms.gdal.drought import SummaryTableDrought

        obj = SummaryTableDrought(
            annual_area_by_drought_class=[{0: 1.0}],
            annual_population_by_drought_class_total=[{0: 1.0}],
            annual_population_by_drought_class_male=[{0: 1.0}],
            annual_population_by_drought_class_female=[{0: 1.0}],
            dvi_value_sum_and_count=(1.0, 1),
        )
        data = obj.dump()
        assert isinstance(data, dict)
        assert "dvi_value_sum_and_count" in data


# ===================================================================
# 4. Schema attribute tests for all algorithm dataclasses
# ===================================================================


class TestSchemaAttributes:
    """Every @marshmallow_dataclass.dataclass should have a callable .Schema()."""

    def test_image_info_schema(self):
        pytest.importorskip("osgeo", reason="GDAL not installed")
        from te_algorithms.gdal.util import ImageInfo

        assert callable(ImageInfo.Schema)
        schema = ImageInfo.Schema()
        assert hasattr(schema, "load")

    def test_summary_table_ld_schema(self):
        from te_algorithms.gdal.land_deg.models import SummaryTableLD

        assert callable(SummaryTableLD.Schema)

    def test_summary_table_ld_status_schema(self):
        from te_algorithms.gdal.land_deg.models import SummaryTableLDStatus

        assert callable(SummaryTableLDStatus.Schema)

    def test_summary_table_ld_change_schema(self):
        from te_algorithms.gdal.land_deg.models import SummaryTableLDChange

        assert callable(SummaryTableLDChange.Schema)

    def test_summary_table_ld_error_recode_schema(self):
        from te_algorithms.gdal.land_deg.models import SummaryTableLDErrorRecode

        assert callable(SummaryTableLDErrorRecode.Schema)

    def test_summary_table_drought_schema(self):
        pytest.importorskip("openpyxl", reason="openpyxl not installed")
        pytest.importorskip("osgeo", reason="GDAL not installed")
        from te_algorithms.gdal.drought import SummaryTableDrought

        assert callable(SummaryTableDrought.Schema)


# ===================================================================
# 5. te_schemas types used inside te_algorithms
# ===================================================================


class TestTeSchemasDependencies:
    """Verify that te_schemas types imported by te_algorithms work correctly."""

    def test_band_serialization(self):
        from te_schemas.results import Band

        b = Band(name="test", metadata={"key": "val"})
        data = Band.Schema().dump(b)
        loaded = Band.Schema().load(data)
        assert loaded.name == "test"

    def test_data_type_enum(self):
        from te_schemas.results import DataType

        assert DataType("Int16") == DataType.INT16
        assert DataType("Float32") == DataType.FLOAT32

    def test_raster_file_type_enum(self):
        from te_schemas.results import RasterFileType

        assert RasterFileType("GeoTiff") == RasterFileType.GEOTIFF

    def test_uri_roundtrip(self):
        from te_schemas.results import URI

        obj = URI(uri="https://example.com/data.tif", etag=None)
        data = URI.Schema().dump(obj)
        loaded = URI.Schema().load(data)
        assert loaded.uri == "https://example.com/data.tif"

    def test_raster_results_roundtrip(self):
        from te_schemas.results import (
            URI,
            Band,
            DataType,
            Raster,
            RasterFileType,
            RasterResults,
        )

        rr = RasterResults(
            name="algo-output",
            rasters={
                "INT16": Raster(
                    uri=URI(uri="https://example.com/r.tif", etag=None),
                    bands=[Band(name="b", metadata={})],
                    datatype=DataType.INT16,
                    filetype=RasterFileType.GEOTIFF,
                )
            },
        )
        data = RasterResults.Schema().dump(rr)
        loaded = RasterResults.Schema().load(data)
        assert loaded.name == "algo-output"
        assert "INT16" in loaded.rasters

    def test_job_schema_loads(self):
        """Job schema from te_schemas must load correctly."""
        from te_schemas.jobs import Job, JobStatus
        import uuid

        job_data = {
            "id": str(uuid.uuid4()),
            "params": {},
            "progress": 100,
            "start_date": "2025-01-01T00:00:00",
            "status": "FINISHED",
            "task_name": "algo test",
        }
        loaded = Job.Schema().load(job_data)
        assert loaded.status == JobStatus.FINISHED

    def test_schema_base_inheritance(self):
        """Algorithm models using SchemaBase should have schema/dump/validate."""
        from te_algorithms.gdal.land_deg.models import SummaryTableLDErrorRecode

        obj = SummaryTableLDErrorRecode(baseline_summary={1: 0.5})
        data = obj.dump()
        assert isinstance(data, dict)
        # validate() should not raise
        obj.validate()

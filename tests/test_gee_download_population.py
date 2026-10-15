from unittest.mock import Mock, patch

from te_algorithms.gee import download


def _mock_worldpop_collection():
    collection = Mock()
    filtered = Mock()
    collection.filterDate.return_value = filtered
    filtered.first.return_value.select.return_value.projection.return_value = (
        "worldpop-projection"
    )

    male_bands = Mock()
    male_renamed = Mock()
    male_bands.rename.return_value = male_renamed
    female_bands = Mock()
    female_renamed = Mock()
    female_bands.rename.return_value = female_renamed
    population_image = Mock()
    male_renamed.addBands.return_value = population_image
    filtered.select.side_effect = lambda band: {
        "male": Mock(toBands=Mock(return_value=male_bands)),
        "female": Mock(toBands=Mock(return_value=female_bands)),
    }[band]
    population_image.divide.return_value.multiply.return_value.reproject.return_value = "worldpop-density"

    return collection, population_image


@patch("te_algorithms.gee.download.ee.Image")
@patch("te_algorithms.gee.download.ee.ImageCollection")
def test_worldpop_download_converts_counts_to_density(image_collection, image_class):
    collection, population_image = _mock_worldpop_collection()
    image_collection.return_value = collection
    image_class.pixelArea.return_value = "pixel-area"

    result = download._get_population(2020, "worldpop", add_to_map=True)

    assert result["image"] == "worldpop-density"
    assert all(
        band.metadata["units"] == "people per hectare" for band in result["bands"]
    )
    population_image.divide.assert_called_once_with("pixel-area")
    population_image.divide.return_value.multiply.assert_called_once_with(10000)
    population_image.divide.return_value.multiply.return_value.reproject.assert_called_once_with(
        "worldpop-projection"
    )


@patch("te_algorithms.gee.download.ee.Image")
@patch("te_algorithms.gee.download.ee.ImageCollection")
def test_gpw_download_converts_counts_to_density(image_collection, image_class):
    collection = Mock()
    filtered = Mock()
    collection.filterDate.return_value = filtered
    image_collection.return_value = collection
    population_image = Mock()
    filtered.select.return_value.mosaic.return_value.rename.return_value = (
        population_image
    )
    filtered.first.return_value.select.return_value.projection.return_value = (
        "gpw-projection"
    )
    population_image.divide.return_value.multiply.return_value.reproject.return_value = "gpw-density"
    image_class.pixelArea.return_value = "pixel-area"

    result = download._get_gpw_population(2020, "gpw", add_to_map=True)

    assert result["image"] == "gpw-density"
    assert result["bands"][0].metadata["type"] == "total"
    assert result["bands"][0].metadata["units"] == "people per hectare"
    population_image.divide.assert_called_once_with("pixel-area")
    population_image.divide.return_value.multiply.return_value.reproject.assert_called_once_with(
        "gpw-projection"
    )

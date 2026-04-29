# trends.earth-algorithms

[![Trends.Earth](https://s3.amazonaws.com/trends.earth/sharing/trends_earth_logo_bl_600width.png)](http://trends.earth)

[![Ruff](https://github.com/ConservationInternational/trends.earth-algorithms/actions/workflows/ruff.yaml/badge.svg)](https://github.com/ConservationInternational/trends.earth-algorithms/actions/workflows/ruff.yaml)
[![Test](https://github.com/ConservationInternational/trends.earth-algorithms/actions/workflows/test.yaml/badge.svg)](https://github.com/ConservationInternational/trends.earth-algorithms/actions/workflows/test.yaml)
[![docs](https://readthedocs.org/projects/trendsearth-algorithms/badge/?version=latest)](https://app.readthedocs.org/projects/trendsearth-algorithms/builds/)

`trends.earth-algorithms` is a Python package to facilitate analyzing
remotely-sensed datasets in support of monitoring land degradation. This
project contains common code used by the scripts in
[trends.earth](https://github.com/ConservationInternational/trends.earth).

`Trends.Earth` is a free and open source tool to understand land change: the how and why
behind changes on the ground. Trends.Earth allows users to draw on the best available
information from across a range of sources — from globally available data to customized
local maps. A broad range of users are applying Trends.Earth for projects ranging from
planning and monitoring restoration efforts, to tracking urbanization, to developing
official national reports for submission to the United Nations Convention to Combat
Desertification (UNCCD).

`Trends.Earth` was produced by a partnership of Conservation International, Lund
University, and the National Aeronautics and Space Administration (NASA), with
the support of the Global Environment Facility (GEF). It was further developed
through a partnership with Conservation International, University of Bern,
University of Colorado in partnership with USDA and USAID, University of California —
Santa Barbara in partnership with University of North Carolina — Wilmington and Brown
University with additional funding from the Global Environment Facility (GEF).

## Documentation

For further information on `trends.earth-algorithms` see
[the documentation](https://trendsearth-algorithms.readthedocs.io).

## Installation

GDAL must be installed before installing `trends.earth-algorithms`. On
Ubuntu/Debian:

```bash
sudo apt-add-repository ppa:ubuntugis/ubuntugis-unstable
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev
```

Then install the package:

```bash
git clone https://github.com/ConservationInternational/trends.earth-algorithms
cd trends.earth-algorithms
pip install -e .
```

## Contributing

Contributions are welcome. Please report bugs or suggest improvements via the
[issue tracker](https://github.com/ConservationInternational/trends.earth-algorithms/issues).

## Related Projects

`Trends.Earth` is built from a set of interconnected repositories:

- [trends.earth](https://github.com/ConservationInternational/trends.earth) — QGIS plugin for land degradation monitoring
- [trends.earth-schemas](https://github.com/ConservationInternational/trends.earth-schemas) — Data schemas for analysis results
- [trends.earth-API](https://github.com/ConservationInternational/trends.earth-API) — Backend REST API
- [trends.earth-Environment](https://github.com/ConservationInternational/trends.earth-Environment) — Job execution environment for running scripts
- [trends.earth-CLI](https://github.com/ConservationInternational/trends.earth-CLI) — Command-line interface for developing custom scripts
- [trends.earth-api-ui](https://github.com/ConservationInternational/trends.earth-api-ui) — Web UI for API management

## License

`trends.earth-algorithms` is free and open-source. MIT License — see [LICENSE](LICENSE).

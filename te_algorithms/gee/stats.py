import ee


def get_kendall_coef(n, level=95):
    # The minus 4 is because the indexing below for a sample size of 4
    assert n >= 4
    n = n - 4
    coefs = {
        90: [
            4,
            6,
            7,
            9,
            10,
            12,
            15,
            17,
            18,
            22,
            23,
            27,
            28,
            32,
            35,
            37,
            40,
            42,
            45,
            49,
            52,
            56,
            59,
            61,
            66,
            68,
            73,
            75,
            80,
            84,
            87,
            91,
            94,
            98,
            103,
            107,
            110,
            114,
            119,
            123,
            128,
            132,
            135,
            141,
            144,
            150,
            153,
            159,
            162,
            168,
            173,
            177,
            182,
            186,
            191,
            197,
            202,
        ],
        95: [
            4,
            6,
            9,
            11,
            14,
            16,
            19,
            21,
            24,
            26,
            31,
            33,
            36,
            40,
            43,
            47,
            50,
            54,
            59,
            63,
            66,
            70,
            75,
            79,
            84,
            88,
            93,
            97,
            102,
            106,
            111,
            115,
            120,
            126,
            131,
            137,
            142,
            146,
            151,
            157,
            162,
            168,
            173,
            179,
            186,
            190,
            197,
            203,
            208,
            214,
            221,
            227,
            232,
            240,
            245,
            251,
            258,
        ],
        99: [
            6,
            8,
            11,
            18,
            22,
            25,
            29,
            34,
            38,
            41,
            47,
            50,
            56,
            61,
            65,
            70,
            76,
            81,
            87,
            92,
            98,
            105,
            111,
            116,
            124,
            129,
            135,
            142,
            150,
            155,
            163,
            170,
            176,
            183,
            191,
            198,
            206,
            213,
            221,
            228,
            236,
            245,
            253,
            260,
            268,
            277,
            285,
            294,
            302,
            311,
            319,
            328,
            336,
            345,
            355,
            364,
        ],
    }
    return coefs[level][n]


def mann_kendall(imageCollection):
    """Calculate Mann Kendall's S statistic.

    This function returns the Mann Kendall's S statistic, assuming that n is
    less than 40. The significance of a calculated S statistic is found in
    table A.30 of Nonparametric Statistical Methods, second edition by
    Hollander & Wolfe.

    Args:
        imageCollection: A Google Earth Engine image collection.

    Returns:
        A Google Earth Engine image collection with Mann Kendall statistic for
            each pixel.
    """
    TimeSeriesList = imageCollection.toList(50)
    NumberOfItems = TimeSeriesList.length().getInfo()
    ConcordantArray = []
    DiscordantArray = []
    for k in range(0, NumberOfItems - 1):
        CurrentImage = ee.Image(TimeSeriesList.get(k))
        for l in range(k + 1, NumberOfItems):
            nextImage = ee.Image(TimeSeriesList.get(l))
            Concordant = CurrentImage.lt(nextImage)
            ConcordantArray.append(Concordant)
            Discordant = CurrentImage.gt(nextImage)
            DiscordantArray.append(Discordant)
    ConcordantSum = ee.ImageCollection(ConcordantArray).sum()
    DiscordantSum = ee.ImageCollection(DiscordantArray).sum()
    MKSstat = ConcordantSum.subtract(DiscordantSum)
    return MKSstat

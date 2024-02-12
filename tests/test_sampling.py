from collections import Counter
import cv2
import json
import numpy
from pathlib import Path
import pytest

from src import sampling


# Fun fact - I tried to make these tuples but json only returns lists
SAMPLES = [
    ["1689083978506208_10_100_2_108.jpg", [105, 631], sampling.CLASSES[0]],
    ["1689083978506208_10_100_2_108.jpg", [1175, 2631], sampling.CLASSES[1]],
    ["1698691127000000_10_100_2_108.jpg", [2316, 603], sampling.CLASSES[1]],
    ["1695313597000000_10_100_2_108.jpg", [892, 2890], sampling.CLASSES[1]],
    ["1695760162000000_10_100_2_108.jpg", [1023, 1890], sampling.CLASSES[0]],
    ["1697058742000000_10_100_2_108.jpg", [2134, 1382], sampling.CLASSES[1]],
    ["1697058742000000_10_100_2_108.jpg", [348, 1238], sampling.CLASSES[0]],
]


def test_create_samples():

    # NOTE: These are BGR images (because cv2)
    blue = numpy.zeros((500, 500, 3), dtype=numpy.uint8)
    blue[47, 195] = (255, 0, 0)
    blue[250, 129] = (255, 0, 0)
    red = numpy.zeros((500, 500, 3), dtype=numpy.uint8)
    red[301, 452] = (0, 0, 255)

    savedir = Path("/tmp/test_create_samples/")
    if not savedir.is_dir():
        savedir.mkdir()
    cv2.imwrite("/tmp/test_create_samples/test_blue.png", blue)
    cv2.imwrite("/tmp/test_create_samples/test_red.png", red)

    results = sampling.create_samples(savedir)
    assert results == [
        ["test_blue.png", [47, 195], sampling.CLASSES[0]],
        ["test_blue.png", [250, 129], sampling.CLASSES[0]],
        ["test_red.png", [301, 452], sampling.CLASSES[1]],
    ]

    json.dumps(results)


@pytest.mark.parametrize("name", ("thing.xyz", "Documents/", "jk.jpg"))
def test_ingest_fail(name):
    with pytest.raises(AssertionError):
        sampling.ingest(Path(name))


def test_ingest():
    path = Path("/tmp/test_ingest.json")
    json.dump(SAMPLES, path.open("w"))
    samples = sampling.ingest(path)
    assert samples == SAMPLES


def test_ingest_even():
    path = Path("/tmp/test_ingest.json")
    json.dump(SAMPLES, path.open("w"))
    samples = sampling.ingest(path, make_even=True)

    # Check that we have an even number of labels
    assert len(samples) % 2 == 0
    counted = Counter(sampling.get_labels(samples))
    for label, count in counted.items():
        assert (count == numpy.array(list(counted.values()))).all()


@pytest.mark.parametrize(
    "samples, downsample, window, results",
    (
        # Basic, window of 3
        (
            [
                ["test_patch.png", [3, 3], None],
                ["test_patch.png", [2, 6], None],
                ["test_patch.png", [7, 4], None],
            ],
            1,
            3,
            [
                {0: 33, 1: 44, 2: 55},
                {0: 26, 2: 48},
                {1: 85, 2: 96},
            ],
        ),
        # Try a window of 5
        (
            [
                ["test_patch.png", [3, 3], None],
                ["test_patch.png", [2, 6], None],
                ["test_patch.png", [6, 4], None],
            ],
            1,
            5,
            [
                {0: 22, 2: 44, 4: 66},
                {0: 15, 2: 37},
                {1: 64, 4: 97},
            ],
        ),
        # Try downsample of 2, odd indices will round up
        (
            [
                ["test_patch.png", [2, 2], None],
                ["test_patch.png", [3, 6], None],
                ["test_patch.png", [5, 3], None],
            ],
            2,
            3,
            [
                {0: 11, 1: 33, 2: 55},
                {0: 35, 1: 57, 2: 79},
                {0: 53, 1: 75, 2: 97},
            ],
        ),
        # Downsample of 3, indices will round (we can only place in the middle)
        (
            [
                ["test_patch.png", [2, 2], None],
                ["test_patch.png", [3, 4], None],
            ],
            3,
            3,
            [
                {0: 11, 1: 44, 2: 77},
                {0: 11, 1: 44, 2: 77},
            ],
        ),
        # Try going off the edge
        (
            [
                ["test_patch.png", [8, 8], None],
                ["test_patch.png", [8, 3], None],
                ["test_patch.png", [0, 6], None],
            ],
            1,
            3,
            [
                {0: 88, 1: 99, 2: 0},
                {0: 83, 1: 94, 2: 0},
                {0: 0, 1: 17, 2: 28},
            ],
        ),
        # Off the edge with more downsampling (odd indices round up)
        (
            [
                ["test_patch.png", [0, 0], None],
                ["test_patch.png", [8, 3], None],
                ["test_patch.png", [0, 7], None],
            ],
            2,
            5,
            [
                {0: 0, 1: 0, 2: 11, 3: 33, 4: 55},
                {0: 51, 1: 73, 2: 95, 3: 0, 4: 0},
                {0: 0, 1: 0, 2: 19, 3: 0, 4: 0},
            ],
        ),
    ),
)
def test_get_patches(samples, downsample, window, results):

    test_im = numpy.array(
        [
            [11, 12, 13, 14, 15, 16, 17, 18, 19],
            [21, 22, 23, 24, 25, 26, 27, 28, 29],
            [31, 32, 33, 34, 35, 36, 37, 38, 39],
            [41, 42, 43, 44, 45, 46, 47, 48, 49],
            [51, 52, 53, 54, 55, 56, 57, 58, 59],
            [61, 62, 63, 64, 65, 66, 67, 68, 69],
            [71, 72, 73, 74, 75, 76, 77, 78, 79],
            [81, 82, 83, 84, 85, 86, 87, 88, 89],
            [91, 92, 93, 94, 95, 96, 97, 98, 99],
        ],
        dtype=numpy.uint8,
    )
    test_im = numpy.dstack([test_im] * 3)
    cv2.imwrite("/tmp/test_patch.png", test_im)

    patches = sampling.get_patches(
        imdir=Path("/tmp/"), samples=samples, downsample=downsample, window=window
    )
    assert len(patches) == len(results)

    for patch in patches:
        assert patch.shape == (window, window, 3)

    for patch, result in zip(patches, results):
        for i, value in result.items():
            assert numpy.allclose(patch[i, i], value)


def test_smart_downsample():
    vectors = numpy.array(
        [
            # Cluster 1: Around [1, 1]
            [0, 0],
            [2, 0],
            [0, 2],
            [2, 2],
            [1.01, 1.01],
            # Cluster 2: Around [10, 10]
            [9, 9],
            [11, 9],
            [9, 11],
            [11, 11],
            [9.99, 10.01],
        ]
    )
    result, indices = sampling.smart_downsample(vectors, 2)
    assert result.shape == (2, 2)

    # We can't control the order, but one of these will be true
    assert numpy.allclose(result[0], [1.01, 1.01]) or numpy.allclose(
        result[1], [1.01, 1.01]
    )
    assert numpy.allclose(result[0], [9.99, 10.01]) or numpy.allclose(
        result[1], [9.99, 10.01]
    )

    assert len(indices) == 2
    assert 4 in indices
    assert 9 in indices

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

    assert sampling.create_samples(savedir) == [
        ["test_blue.png", [47, 195], sampling.CLASSES[0]],
        ["test_blue.png", [250, 129], sampling.CLASSES[0]],
        ["test_red.png", [301, 452], sampling.CLASSES[1]],
    ]


@pytest.mark.parametrize("name", ("thing.xyz", "Documents/", "jk.jpg"))
def test_ingest_fail(name):
    with pytest.raises(AssertionError):
        sampling.ingest(Path(name))


def test_ingest():
    path = Path("/tmp/test_ingest.json")
    json.dump(SAMPLES, path.open("w"))
    samples = sampling.ingest(path)
    assert samples == SAMPLES

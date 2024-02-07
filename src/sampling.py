"""
Helper functions to deal with samples
"""

import cv2
import json
import numpy
from pathlib import Path


CLASSES = ("not-plant", "plant")


def create_samples(imdir):
    """
    From a directory of images, build the sample file based on specifically
    colored pixels. Blue=not-plant, Red=plant.
    """
    samples = []
    for impath in sorted(imdir.glob("*.png")):
        bgr = cv2.imread(str(impath))
        blue = numpy.all(bgr == (255, 0, 0), axis=2)
        red = numpy.all(bgr == (0, 0, 255), axis=2)
        if numpy.any(blue):
            for i, j in zip(*numpy.where(blue)):
                samples.append([impath.name, [i, j], "not-plant"])
        if numpy.any(red):
            for i, j in zip(*numpy.where(red)):
                samples.append([impath.name, [i, j], "plant"])
    return samples


def ingest(path):
    assert path.name.endswith(".json"), f"{path} must be json"
    return json.load(path.open("r"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "imdir",
        help="Path to labelled images",
        type=Path,
    )
    parser.add_argument(
        "save",
        help="JSON path to where we want to save the samples",
        type=Path,
    )
    args = parser.parse_args()

    assert args.imdir.is_file(), f"{args.imdir.absolute()} is not a file"

    samples = create_samples(imdir)
    json.dump(samples, save.open("w"), indent=4)

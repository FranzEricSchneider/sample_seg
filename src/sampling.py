"""
Helper functions to deal with samples
"""

import argparse
from collections import Counter, defaultdict
import cv2
import json
import numpy
from pathlib import Path
from sklearn.cluster import KMeans


CLASSES = ("not-plant", "plant")
LABELS = {
    CLASSES[0]: 0,
    CLASSES[1]: 1,
}


def create_samples(imdir, rename=None):
    """
    From a directory of images, build the sample file based on specifically
    colored pixels. Blue=not-plant, Red=plant.
    """
    samples = []
    for impath in sorted(imdir.glob("*.png")):

        if rename is not None:
            name = impath.name.replace(".png", rename)
        else:
            name = impath.name

        bgr = cv2.imread(str(impath))
        blue = numpy.all(bgr == (255, 0, 0), axis=2)
        red = numpy.all(bgr == (0, 0, 255), axis=2)
        if numpy.any(blue):
            for i, j in zip(*numpy.where(blue)):
                samples.append([name, [int(i), int(j)], "not-plant"])
        if numpy.any(red):
            for i, j in zip(*numpy.where(red)):
                samples.append([name, [int(i), int(j)], "plant"])

    return samples


def ingest(path, make_even=False):
    assert path.name.endswith(".json"), f"{path} must be json"
    samples = json.load(path.open("r"))
    if make_even:
        labels = numpy.array(get_labels(samples))
        counted = Counter(labels)
        least = min(counted.values())
        include = []
        for label in set(labels):
            if counted[label] == least:
                include.extend(numpy.where(labels == label)[0].tolist())
            else:
                choices = numpy.where(labels == label)[0]
                include.extend(
                    numpy.random.choice(choices, size=least, replace=False).tolist()
                )
        return [samples[i] for i in sorted(include)]
    else:
        return samples


def get_patches(imdir, samples, downsample, window):

    assert window % 2 == 1, f"Window ({window}) should be odd"
    assert isinstance(downsample, int), f"Downsample ({downsample}) should be int"
    assert downsample > 0, f"Downsample ({downsample}) should be > 0"

    # Figure out which images we want to draw from
    sampled_ims = defaultdict(list)
    for i, sample in enumerate(samples):
        sampled_ims[sample[0]].append(i)

    patches = []
    for imname, sample_ids in sampled_ims.items():
        impath = imdir.joinpath(imname)
        assert impath.is_file(), f"Sampled file {impath.absolute()} not found"
        rgb = cv2.cvtColor(cv2.imread(str(impath)), cv2.COLOR_BGR2RGB)

        if downsample > 1:
            rgb = rgb[::downsample, ::downsample]

        # Pad to avoid extracting off the edge
        # pad_width: ((before/after), (before/after), (before/after))
        rgb = numpy.pad(
            rgb,
            pad_width=((window,) * 2,) * 2 + ((0, 0),),
            mode="constant",
            constant_values=0,
        )

        for i in sample_ids:
            _, pixel, label = samples[i]

            if downsample > 1:
                pixel = ((numpy.array(pixel) / downsample) + 0.5).astype(int)

            # The extra (+window) term is to account for the padding
            ipix = pixel[0] - (window // 2) + window
            jpix = pixel[1] - (window // 2) + window

            patches.append(rgb[ipix : ipix + window, jpix : jpix + window].copy())

    return patches


def get_labels(samples):
    return [LABELS[sample[-1]] for sample in samples]


def smart_downsample(vectors, target_size):
    kmeans = KMeans(n_clusters=target_size, n_init="auto")
    kmeans.fit(vectors)
    centroids = kmeans.cluster_centers_
    indices = [
        numpy.argmin(numpy.linalg.norm(vectors - centroid, axis=1))
        for centroid in centroids
    ]
    return numpy.array([vectors[i] for i in indices]), indices


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

    assert args.imdir.is_dir(), f"{args.imdir.absolute()} is not a directory"

    samples = create_samples(args.imdir, rename=".jpg")
    json.dump(samples, args.save.open("w"), indent=4)

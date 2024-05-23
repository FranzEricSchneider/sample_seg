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
from sklearn.model_selection import train_test_split


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
        blue = numpy.all(bgr == (255, 38, 0), axis=2)
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


def split_by_image(samples, val_size, test_size):

    # Split the images up into different groups
    imnames = sorted(set([sample[0] for sample in samples]))
    train_imnames, temp_imnames = train_test_split(
        imnames,
        test_size=(val_size + test_size),
        random_state=42,
    )
    val_imnames, test_imnames = train_test_split(
        temp_imnames,
        test_size=test_size / (val_size + test_size),
        random_state=42,
    )

    # Save these names so other tests can potentially reference them
    path = Path("/tmp/sample_seg_file_breakdown.json")
    json.dump(
        {
            "train": train_imnames,
            "val": val_imnames,
            "test": test_imnames,
        },
        path.open("w"),
        indent=4,
        sort_keys=True,
    )
    print(f"Saved train/val/test split to {path}")

    # Then bookkeep which samples correspond to which images
    def allocate(names):
        return [sample for sample in samples if sample[0] in names]

    return allocate(train_imnames), allocate(val_imnames), allocate(test_imnames)


def get_patches(imdir, samples, downsample, windows):

    for window in windows:
        assert window % 2 == 1, f"Windows ({windows}) should be odd"
    assert len(windows) == len(set(windows)), f"windows ({windows}) must be unique"

    assert isinstance(downsample, int), f"Downsample ({downsample}) should be int"
    assert downsample > 0, f"Downsample ({downsample}) should be > 0"

    # Figure out which images we want to draw from
    sampled_ims = defaultdict(list)
    for i, sample in enumerate(samples):
        sampled_ims[sample[0]].append(i)

    # Track the patches that we collect and the original samples indices those
    # patches correspond to (for eventual reordering)
    patches = defaultdict(list)
    indices = []

    max_window = max(windows)

    for imname, sample_ids in sorted(sampled_ims.items()):
        impath = imdir.joinpath(imname)
        assert impath.is_file(), f"Sampled file {impath.absolute()} not found"
        rgb = cv2.cvtColor(cv2.imread(str(impath)), cv2.COLOR_BGR2RGB)

        if downsample > 1:
            rgb = rgb[::downsample, ::downsample]

        # Pad to avoid extracting off the edge
        # pad_width: ((before/after), (before/after), (before/after))
        rgb = numpy.pad(
            rgb,
            pad_width=((max_window,) * 2,) * 2 + ((0, 0),),
            mode="constant",
            constant_values=0,
        )

        for i in sample_ids:
            _, pixel, _ = samples[i]

            if downsample > 1:
                pixel = ((numpy.array(pixel) / downsample) + 0.5).astype(int)

            for window in windows:
                # The extra (+max_window) term is to account for the padding
                ipix = pixel[0] - (window // 2) + max_window
                jpix = pixel[1] - (window // 2) + max_window

                patches[window].append(
                    rgb[ipix : ipix + window, jpix : jpix + window].copy()
                )

            indices.append(i)

    # Re-sort the patches into the original sample order
    order = numpy.argsort(indices)
    return {window: [patches[window][i] for i in order] for window in windows}


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

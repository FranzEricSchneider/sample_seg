import argparse
import cv2
import joblib
import json
import numpy
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from compare import get_featurators
from masks import rgb01
from sampling import get_patches


class StitchKNN:
    def __init__(self, frequency, k):
        self.f = frequency
        self.knn = KNeighborsClassifier(n_neighbors=k)

    def pixels(self, impath, downsample):
        """
        NOTE: These in the frame of the full image, you need to downsample them
        separately if you want the downsampled positions
        """
        size = cv2.imread(str(impath)).shape[:2]
        for i in range(0, size[0], self.f * downsample):
            for j in range(0, size[1], self.f * downsample):
                yield [i, j]

    def stitch(self, impath, downsample, pixels, labels):
        size = cv2.imread(str(impath)).shape[:2]
        size = (size[0] // downsample, size[1] // downsample)

        # Fit the classifier on the sample data
        self.knn.fit(pixels, labels)

        # Make an (N, 2) array of all the pixels we want to fill in and predict
        grid_i, grid_j = numpy.meshgrid(range(size[0]), range(size[1]), indexing="ij")
        pixels = numpy.c_[grid_i.ravel(), grid_j.ravel()]
        labels = self.knn.predict(pixels)

        # Convert to the image shape and make it boolean
        return (labels > 0.5).reshape(size)


STITCHERS = {
    "knn": StitchKNN,
}


def main(imdir, cpath, mosdir, savedir, sname, kwargs):

    # Load the classifier
    classifier = joblib.load(cpath)
    fname, _, downsample = cpath.stem.split(":")
    downsample = int(downsample)

    # Choose
    stitcher = STITCHERS[sname](**kwargs)

    # Load featurator
    featurators, _ = get_featurators(mosdir)
    featurator = featurators[fname]
    windows = [f.window for f in featurator]

    all_samples = []

    for impath in tqdm(sorted(imdir.glob("*.jpg"))):

        # Mimic the hand-labeled sample format (imname, [pixel], label), just leave
        # the label as None
        samples = [
            (impath.name, pixel, None) for pixel in stitcher.pixels(impath, downsample)
        ]

        # Extract snippets from the images
        patches = get_patches(imdir, samples, downsample, windows)

        # Turn those into vectors and process them
        vectors = numpy.hstack(
            [fset.transform(patches[fset.window]) for fset in featurator]
        )
        predicted = classifier.predict(vectors)

        mask = stitcher.stitch(
            impath,
            downsample,
            numpy.array([pixel for _, pixel, _ in samples]) / downsample,
            predicted,
        )
        numpy.save(savedir / f"{impath.stem}.npy", mask)

        for (imname, pixel, _), label in zip(samples, predicted):
            all_samples.append((imname, pixel, "plant" if label == 1 else "not-plant"))

    json.dump(
        all_samples,
        (savedir / "samples.json").open("w"),
        indent=4,
        sort_keys=True,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "imdir",
        help="Path to the directory with the jpg images we want to process",
        type=Path,
    )
    parser.add_argument(
        "classifier_path",
        help="Path to trained classifier pkl with format fname:cname:downsample",
        type=Path,
    )
    parser.add_argument(
        "mosaik_dir",
        help="Directory where MOSAIKS featurators can be found",
        type=Path,
    )
    parser.add_argument(
        "save_dir",
        help="Directory where to save the converted npy masks",
        type=Path,
    )
    parser.add_argument(
        "-k",
        "--stitcher-kwargs",
        help="json encoded string with stitcher kwargs",
        default=json.dumps({"frequency": 8, "k": 3}),
    )
    parser.add_argument(
        "-s",
        "--stitcher-choice",
        help=f"Choice of stitcher ({STITCHERS.keys()})",
        default="knn",
    )
    args = parser.parse_args()

    main(
        imdir=args.imdir,
        cpath=args.classifier_path,
        mosdir=args.mosaik_dir,
        savedir=args.save_dir,
        sname=args.stitcher_choice,
        kwargs=json.loads(args.stitcher_kwargs),
    )

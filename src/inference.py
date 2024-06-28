import argparse
import cProfile
import cv2
import joblib
import json
import numpy
from pathlib import Path
from skimage import io
from sklearn.neighbors import KNeighborsClassifier
from skimage.segmentation import slic
from skimage.util import img_as_float
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
        propagated_labels = self.knn.predict(pixels)

        # Convert to the image shape and make it boolean
        return (propagated_labels > 0.5).reshape(size)


class StitchSLIC:
    def __init__(self, num_segments, sigma):
        self.n = num_segments
        self.sigma = sigma
        self.indices = None

    def pixels(self, impath, downsample):
        """
        NOTE: These in the frame of the full image, you need to downsample them
        separately if you want the downsampled positions
        """
        image = img_as_float(io.imread(impath))[::downsample, ::downsample]

        # The first time through, populate the indices based on image size
        if self.indices is None:
            self.indices = numpy.dstack(
                numpy.meshgrid(
                    range(image.shape[0]), range(image.shape[1]), indexing="ij"
                )
            )

        self.segments = slic(image, n_segments=self.n, sigma=self.sigma)
        self.ids = sorted(numpy.unique(self.segments))

        for segid in self.ids:
            mask = self.segments == segid
            yield (closest_pixel_to_center(self.indices[mask]) * downsample).tolist()

    def stitch(self, impath, downsample, pixels, labels):
        mask = numpy.zeros(self.segments.shape, dtype=bool)
        for segid, label in zip(self.ids, labels):
            if label:
                mask[self.segments == segid] = True
        return mask


def closest_pixel_to_center(indices):

    # Compute the distances from the center of mass and get the smallest
    center_of_mass = indices.mean(axis=0)
    distances = numpy.linalg.norm(indices - center_of_mass, axis=1)
    closest = numpy.argmin(distances)

    # Return the coordinates of the closest True pixel
    return indices[closest]


STITCHERS = {
    "knn": StitchKNN,
    "slic": StitchSLIC,
}


def main(imdir, cpath, mosdir, savedir, sname, stitch_downsample, kwargs):

    # Load the classifier
    classifier = joblib.load(cpath)
    fname, _, seg_downsample = cpath.stem.split(":")
    seg_downsample = int(seg_downsample)

    # Choose
    stitcher = STITCHERS[sname](**kwargs)

    # Load featurator
    featurators, _ = get_featurators(mosdir)
    featurator = featurators[fname]
    windows = [f.window for f in featurator]

    all_samples = []

    for impath in tqdm(sorted(imdir.glob("*.jpg"))):

        # Mimic the hand-labeled sample format (imname, [pixel], label), just
        # leave the label as None. By design these samples are in the frame of
        # the full image
        samples = [
            (impath.name, pixel, None)
            for pixel in stitcher.pixels(impath, stitch_downsample)
        ]

        # Extract snippets from the images
        patches = get_patches(imdir, samples, seg_downsample, windows)

        # Turn those into vectors and process them
        vectors = numpy.hstack(
            [fset.transform(patches[fset.window]) for fset in featurator]
        )
        predicted = classifier.predict(vectors)

        mask = stitcher.stitch(
            impath,
            stitch_downsample,
            numpy.array([pixel for _, pixel, _ in samples]) / stitch_downsample,
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
        "-d",
        "--stitch-downsample",
        help="Downsampling for the stitched image (may affect stitch process,"
        " but is handled separately from classification)",
        default=4,
        type=int,
    )
    parser.add_argument(
        "-k",
        "--stitcher-kwargs",
        help='json encoded string with stitcher kwargs, e.g. \'{"num_segments": 10000, "sigma": 5}\'',
        default=json.dumps({"frequency": 8, "k": 3}),
    )
    parser.add_argument(
        "-s",
        "--stitcher-choice",
        help=f"Choice of stitcher ({STITCHERS.keys()})",
        default="knn",
    )
    args = parser.parse_args()

    profile = cProfile.Profile()
    profile.enable()

    main(
        imdir=args.imdir,
        cpath=args.classifier_path,
        mosdir=args.mosaik_dir,
        savedir=args.save_dir,
        sname=args.stitcher_choice,
        stitch_downsample=args.stitch_downsample,
        kwargs=json.loads(args.stitcher_kwargs),
    )

    profile.disable()
    path = args.save_dir / "profile.snakeviz"
    profile.dump_stats(path)
    print(f"Saved runtime profile information to {path}")

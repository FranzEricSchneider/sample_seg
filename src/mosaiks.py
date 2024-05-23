"""
A basic, partial representation of the patch-based featurization from
    A Generalizable and Accessible Approach to Machine Learning with Global
    Satellite Imagery
Found here: https://www.nature.com/articles/s41467-021-24638-z
"""

import argparse
from collections import Counter
import cv2
from matplotlib import pyplot
import numpy
from pathlib import Path
from sklearn.cluster import KMeans
from tqdm import tqdm

import distinctipy

from zca import ZCA


PATCHES = "patches.npy"
ZCA_MEAN = "zca_mean.npy"
ZCA_WHITEN = "zca_whiten.npy"
ZCA_DEWHITEN = "zca_dewhiten.npy"


class Mosaiks:
    def __init__(self, savedir):
        self.features = numpy.load(savedir.joinpath(PATCHES))

        self.zca = ZCA()
        self.zca.mean_ = numpy.load(savedir.joinpath(ZCA_MEAN))
        self.zca.whiten_ = numpy.load(savedir.joinpath(ZCA_WHITEN))
        self.zca.dewhiten_ = numpy.load(savedir.joinpath(ZCA_DEWHITEN))

        self.m = int(numpy.sqrt(self.features.shape[1] / 3) + 0.5)
        self.patch_size = (self.m,) * 2

        # Renaming for external use
        self.window = self.m

        # Prepare the features by baking in the whitening (mentioned in
        # footnote 14 of the MOSAIKS supplementals)
        # https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-021-24638-z/MediaObjects/41467_2021_24638_MOESM1_ESM.pdf
        self.prewhite_features = self.features @ self.zca.whiten_

    def process_images(self, impaths, downsample=1):
        for impath in tqdm(impaths):
            rgb = rgbim(impath, downsample)
            yield (
                impath,
                self.apply(flatten_image(rgb, self.patch_size)).reshape(
                    (
                        rgb.shape[0] - self.m + 1,
                        rgb.shape[1] - self.m + 1,
                        -1,
                    )
                ),
            )

    def transform(self, patches):
        return self.apply(flatten_patches(patches))

    def apply(self, view):
        """view: 27xM convolutional view into the image"""
        return numpy.dot(
            self.prewhite_features,
            view - self.zca.mean_.reshape((-1, 1)),
        ).T


def rgbim(impath, downsample=1):
    rgb = cv2.cvtColor(cv2.imread(str(impath)), cv2.COLOR_BGR2RGB)
    if downsample > 1:
        rgb = rgb[::downsample, ::downsample]
    return rgb


def create_feature_set(impaths, savedir, m=3, k=512, downsample=1):
    """
    impaths: Iterable containing paths to draw from
    savedir: Where to save feature set and whitening values
    m: Side length of the patches
    k: Number of patches (half positive, half negative)
    """

    assert k % 2 == 0, "Number of patches must be even"
    # Sample each patch as a negative and positive version, so we only need to
    # sample k//2 times
    k2 = k // 2

    indices = numpy.random.randint(0, len(impaths), size=k2)
    patches = []

    for i, number in tqdm(Counter(indices).items()):

        rgb = rgbim(impaths[i], downsample)

        for _ in range(number):
            ipix = numpy.random.randint(0, rgb.shape[0] - m)
            jpix = numpy.random.randint(0, rgb.shape[1] - m)
            # Flatten the patches (e.g. 3x3x3 becomes 1x27), we will need to
            # imitate a convolution using flattened (and correctly indexed)
            # image values
            patch = rgb[ipix : ipix + m, jpix : jpix + m, :].flatten()
            patches.append(patch)
            patches.append(-1 * patch)

    patches = numpy.array(patches)

    # Whiten (center and decorrelate) the patches
    whitener = ZCA()
    whitener.fit(patches)
    patches = whitener.transform(patches)

    def save_npy(path, values):
        numpy.save(path, values)
        print(f"Saved to {path}")

    save_npy(savedir.joinpath(PATCHES), patches)
    save_npy(savedir.joinpath(ZCA_MEAN), whitener.mean_)
    save_npy(savedir.joinpath(ZCA_WHITEN), whitener.whiten_)
    save_npy(savedir.joinpath(ZCA_DEWHITEN), whitener.dewhiten_)


def flatten_image(image, patch_size):
    """
    Flatten an image such that a dot product is the same as convolution
    """
    return numpy.array(
        [
            image[i : i + patch_size[0], j : j + patch_size[1], :].flatten()
            for i in range(image.shape[0] - patch_size[0] + 1)
            for j in range(image.shape[1] - patch_size[1] + 1)
        ]
    ).T


def flatten_patches(patches):
    """
    Flatten a list of patches (window, window, 3) such that a dot product is
    the same as convolution
    """
    return numpy.array([patch.flatten() for patch in patches]).T


def visualize_knn(vpath, k=4):
    vectors = numpy.load(vpath)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")

    m, n, c = vectors.shape
    labels = kmeans.fit_predict(vectors.reshape(m * n, c)).reshape(m, n)
    display = numpy.zeros((m, n, 3))
    for label, color in enumerate(distinctipy.get_colors(k)):
        display[numpy.where(labels == label)] = color
    pyplot.imshow(display)
    path = vpath.with_suffix(".png")
    assert (
        not path.is_file()
    ), f"File {path} already exists and would have been overwritten"
    pyplot.savefig(path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "imdir",
        help="Path to the directory with all of the original images",
        type=Path,
    )
    parser.add_argument(
        "savedir",
        help="Path to the directory where we want to save the created files",
        type=Path,
    )
    parser.add_argument(
        "-d",
        "--downsample",
        help="Downsample value when creating the features",
        type=int,
        default=8,
    )
    parser.add_argument(
        "-k",
        "--channels",
        help="Number of feature channels to have",
        type=int,
        default=128,
    )
    parser.add_argument(
        "-v",
        "--visualize",
        help="Whether or not to visualize some of the images (slow)",
        action="store_true",
    )
    args = parser.parse_args()

    assert args.imdir.is_dir(), f"{args.imdir.absolute()} is not a directory"

    impaths = sorted(args.imdir.glob("*jpg"))
    assert len(impaths) > 0, "Found no *jpg images"

    if not args.savedir.is_dir():
        args.savedir.mkdir()
    create_feature_set(
        impaths, args.savedir, downsample=args.downsample, k=args.channels
    )

    if args.visualize:
        M = Mosaiks(args.savedir)
        indices = numpy.random.randint(0, len(impaths), size=5)
        for impath, vectorized in M.process_images(
            [impaths[i] for i in indices], downsample=args.downsample
        ):
            numpy.save(impath.with_suffix(".npy"), vectorized)
            visualize_knn(impath.with_suffix(".npy"))

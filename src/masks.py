"""
Tools for examining masks and their quality.
"""

import argparse
import cv2
import numpy
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from skimage.transform import resize
import tempfile

from sampling import ingest


def rgb01(impath):
    return cv2.cvtColor(cv2.imread(str(impath)), cv2.COLOR_BGR2RGB) / 255


def uint8(image, bgr=False):
    image = (image * 255).astype(numpy.uint8)
    if bgr:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        return image


def load(mpath, size=(3036, 4024)):
    return resize(
        numpy.load(mpath), size, order=0, preserve_range=True, anti_aliasing=False
    ).astype(bool)


def score(mpaths, samples):
    actual = []
    predicted = []
    for mpath in mpaths:
        # The samples contain (imname, [pixel ij], plant/not-plant)
        relevant = [sample for sample in samples if sample[0].startswith(mpath.stem)]
        mask = load(mpath)
        # Add the scores for this image
        actual.append([plant == "plant" for _, _, plant in relevant])
        predicted.append([mask[pixel[0], pixel[1]] for _, pixel, _ in relevant])

    # Concatenate all the results
    actual = sum(actual, start=[])
    predicted = sum(predicted, start=[])
    return {
        "accuracy": accuracy_score(actual, predicted),
        "F1": f1_score(actual, predicted, average="weighted"),
        "actual": actual,
        "predicted": predicted,
    }


def mask_rgb(mask):
    mask_im = numpy.ones(mask.shape + (3,)) * [0, 0.5, 0]
    mask_im[~mask] = 0
    return mask_im


def color_area(image, i, j, radius, color):
    image[i - radius : i + radius, j - radius : j + radius] = color


def visualize(savedir, imdir, mpaths, samples):

    for mpath in mpaths:

        # The samples contain (imname, [pixel ij], plant/not-plant)
        relevant = [sample for sample in samples if sample[0].startswith(mpath.stem)]
        mask = load(mpath)
        image = rgb01(imdir / (mpath.stem + ".jpg"))
        mask_im = mask_rgb(mask)

        # Highlight the labeled spots on the original image
        im_high = image.copy()
        for _, pixel, plant in relevant:
            color_area(
                im_high, *pixel, 10, [0, 1, 0] if plant == "plant" else [1, 0, 0.6]
            )

        # Highligh the labeled spots on the mask
        m_high = mask_im.copy()
        for _, pixel, plant in relevant:
            # Check whether the mask matches the label
            correct = (plant == "plant") == mask[pixel[0], pixel[1]]
            color_area(im_high, *pixel, 20, [1, 1, 1] if correct else [0, 0, 0])
            color_area(
                im_high, *pixel, 10, [0, 1, 0] if plant == "plant" else [1, 0, 0.6]
            )

        mdir = savedir / f"{mpath.stem}"
        mdir.mkdir()
        for i, save_im in enumerate([image, mask_im, im_high, m_high]):
            cv2.imwrite(str(mdir / f"{i:02}.jpg"), uint8(save_im, bgr=True))
        print(f"Saved to {mdir}")


def main():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "imdir",
        help="Path to the directory with all of the original images",
        type=Path,
    )
    parser.add_argument(
        "maskdir",
        help="Path to the npy files containing masks we want to check out",
        type=Path,
    )
    parser.add_argument(
        "samples",
        help="Path to collected sample json file",
        type=Path,
    )
    parser.add_argument(
        "-s",
        "--savedir",
        help="Directory to save visualizations in",
        type=Path,
        default=Path("/tmp/"),
    )
    args = parser.parse_args()

    mpaths = sorted(args.maskdir.glob("*.npy"))
    samples = ingest(args.samples, make_even=True)

    results = score(mpaths, samples)
    print(f"Accuracy score: {results['accuracy']}")
    print(f"F1 score: {results['F1']}")

    visualize(args.savedir, args.imdir, mpaths, samples)


if __name__ == "__main__":
    main()

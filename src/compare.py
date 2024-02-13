"""
File for high-level tests of the lower-level components
"""

import argparse
from matplotlib import pyplot
import numpy
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from colors import RG, RGB, Gray
from mosaiks import Mosaiks
from nn import EfficientNetB1, ResNet50
from sampling import get_patches, get_labels, ingest, smart_downsample
from vis import random_patch_vis, vector_vis


def save_confusion(cname, stats):
    cm = confusion_matrix(stats["actual"], stats["predicted"], labels=[0, 1])
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    display.plot(cmap="binary")
    pyplot.tight_layout()
    pyplot.savefig(f"/tmp/{cname.replace(' ', '-').replace('/', '_')}.png")
    pyplot.close()


def rank_stat(results, stat_name, top_x=None, text_values=False):

    # Sort the dictionary by average error values
    sorted_results = sorted(results.items(), key=lambda x: x[1][stat_name])

    # Extract names and corresponding stats
    names, stats = zip(*sorted_results)
    stats = [result[stat_name] for result in stats]

    if top_x is not None:
        names = names[:top_x]
        stats = stats[:top_x]

    # Create a bar graph
    figure = pyplot.figure(figsize=(20, 12))
    pyplot.bar(names, stats)
    pyplot.ylim(0.9 * min(stats), 1.1 * max(stats))

    if text_values:
        for name, point in zip(names, stats):
            pyplot.text(name, point, f"{point:.3f}", ha="center", va="bottom")
        pyplot.ylim(0, 1.2 * max(stats))

    pyplot.ylabel(stat_name)
    pyplot.title(f"{stat_name} over treatments")
    pyplot.xticks(rotation=90, ha="right")
    pyplot.tight_layout()
    path = f"/tmp/ranked_{stat_name}.png"
    pyplot.savefig(path)
    pyplot.close()
    print(f"Saved {path}")


def supervised_test(
    imdir, sample_path, mosdir, pca_vis=False, patch_images=False, test_size=0.3
):

    # Ingest samples, throwing away random samples until the class numbers are
    # evenly split
    samples = ingest(sample_path, make_even=True)

    train_samples, test_samples = train_test_split(
        samples,
        test_size=test_size,
        random_state=42,
    )

    featurators = {
        "ResNet": ResNet50(),
        "EfficientNet": EfficientNetB1(),
        f"RG-1": RG(window=1),
    }
    for w in [1, 5]:
        featurators[f"RGB-{w}"] = RGB(window=w)
        featurators[f"Gray-{w}"] = Gray(window=w)

    # for d in [4, 8, 16]:
    #     for k in [64, 128]:
    for d in [1, 4, 8, 16]:
        for k in [64, 128, 512]:
            key = f"M-D{d}-{k}"
            featurators[key] = Mosaiks(mosdir.joinpath(key))

    classifiers = {
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=50),
        "RBF SVM": SVC(kernel="rbf"),
        "Linear SVM": SVC(kernel="linear"),
    }
    # NOTE: Linear SVMs take a long time to fit on many data points. Downsample
    # the points for them somehow.
    classifier_limits = {
        "Linear SVM": 200,
    }

    downsamplings = [1, 2, 4, 8]
    results = {}

    # Initialize tqdm with the total number of iterations
    progress_bar = tqdm(
        total=len(downsamplings) * len(featurators) * len(classifiers),
        desc="Processing",
    )

    for downsample in downsamplings:

        for fname, featurator in featurators.items():

            patches = {}
            vectors = {}
            for name, samples in [("train", train_samples), ("test", test_samples)]:
                patches[name] = get_patches(
                    imdir=imdir,
                    samples=samples,
                    downsample=downsample,
                    window=featurator.window,
                )
                vectors[name] = featurator.transform(patches[name])

            if pca_vis:
                vector_vis(
                    vectors=vectors["train"],
                    labels=get_labels(train_samples),
                    savedir=Path("/tmp"),
                    name=f"{fname}_D-{downsample}",
                )

            for cname, classifier in classifiers.items():

                train_labels = get_labels(train_samples)
                train_vectors = vectors["train"]
                if cname in classifier_limits:
                    train_vectors, indices = smart_downsample(
                        train_vectors,
                        classifier_limits[cname],
                    )
                    train_labels = [train_labels[i] for i in indices]

                # Train
                classifier.fit(train_vectors, train_labels)

                # And test
                actual = get_labels(test_samples)
                predicted = classifier.predict(vectors["test"])
                results[f"{fname}/{cname}/{downsample}"] = {
                    "accuracy": accuracy_score(actual, predicted),
                    "F1": f1_score(actual, predicted, average="weighted"),
                    "actual": actual,
                    "predicted": predicted,
                }

                if patch_images:
                    random_patch_vis(
                        patches=patches["test"],
                        actual=actual,
                        predicted=predicted,
                        number=(10, 16),
                        savedir=Path("/tmp"),
                        name=f"{fname}_D-{downsample}_{cname}",
                    )

                # Update the progress bar by one step
                progress_bar.update(1)

    progress_bar.close()

    for cname, stats in sorted(results.items()):
        print(f"{cname:<30} accuracy: {stats['accuracy']:.3f}, F1: {stats['F1']:.3f}")
        save_confusion(cname, stats)

    rank_stat(results, "accuracy")


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
        "samples",
        help="Path to collected sample json file",
        type=Path,
    )
    parser.add_argument(
        "-m",
        "--mosaic-dir",
        help="Directory where a MOSAIKS featurator can be found",
        type=Path,
    )
    parser.add_argument(
        "-p",
        "--pca-vis",
        help="Whether to run vis of each vector set into /tmp/. Somewhat slow",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--patch-images",
        help="Whether to run vis of patches into /tmp/. Somewhat slow",
        action="store_true",
    )
    args = parser.parse_args()

    assert args.imdir.is_dir(), f"{args.imdir.absolute()} is not a directory"
    assert args.samples.is_file(), f"{args.samples.absolute()} is not a file"

    from cProfile import Profile

    profile = Profile()
    profile.enable()

    supervised_test(
        imdir=args.imdir,
        sample_path=args.samples,
        mosdir=args.mosaic_dir,
        pca_vis=args.pca_vis,
        patch_images=args.patch_images,
    )

    profile.disable()
    profile.dump_stats("/tmp/compare.snakeviz")

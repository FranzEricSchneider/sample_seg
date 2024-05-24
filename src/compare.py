"""
File for high-level tests of the lower-level components
"""

import argparse
from collections import defaultdict
from cProfile import Profile
import joblib
import json
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import time
from tqdm import tqdm

from colors import RG, RGB, Gray
from fourier import Fourier
from mosaiks import Mosaiks
from nn import EfficientNetB1, ResNet50
from sampling import get_patches, get_labels, ingest, smart_downsample, split_by_image
from vis import random_patch_vis, vector_vis


def save_confusion(cname, stats):
    cm = confusion_matrix(stats["actual"], stats["predicted"], labels=[0, 1])
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    display.plot(cmap="binary")
    pyplot.tight_layout()
    pyplot.savefig(f"/tmp/{cname.replace(' ', '-').replace('/', '_')}.png")
    pyplot.close()


def rank_stat(
    results, stat_name, bottom_x=None, top_x=None, text_values=False, suffix=""
):

    # Sort the dictionary by average error values
    sorted_results = sorted(results.items(), key=lambda x: x[1][stat_name])

    # Extract names and corresponding stats
    names, stats = zip(*sorted_results)
    stats = [result[stat_name] for result in stats]

    if bottom_x is not None:
        names = names[:bottom_x]
        stats = stats[:bottom_x]
    if top_x is not None:
        names = names[-top_x:]
        stats = stats[-top_x:]

    # Create a bar graph
    figure = pyplot.figure(figsize=(40, 12))
    pyplot.bar(names, stats)
    pyplot.yticks(numpy.arange(0, 1.05, 0.05))
    pyplot.ylim(0.9 * min(stats), min(1.0, 1.1 * max(stats)))

    if text_values:
        for name, point in zip(names, stats):
            pyplot.text(name, point, f"{point:.3f}", ha="center", va="bottom")
        pyplot.ylim(0, 1.2 * max(stats))

    metatext = ""
    if bottom_x is not None:
        metatext += f"_bottom_{bottom_x}"
    if top_x is not None:
        metatext += f"_top_{top_x}"

    pyplot.ylabel(stat_name)
    pyplot.title(f"{stat_name} over treatments")
    pyplot.xticks(rotation=90, ha="right")
    pyplot.grid(axis="y")
    pyplot.tight_layout()
    path = f"/tmp/ranked_{stat_name}{metatext}{suffix}.png"
    pyplot.savefig(path)
    pyplot.close()
    print(f"Saved {path}")

    return names


def get_featurators(mosdir):
    """Get the common set we want to test"""

    featurators = {
        "ResNet": [ResNet50()],
        "EfficientNet": [EfficientNetB1()],
    }

    for w in [1, 5, 17]:
        featurators[f"RG-{w}"] = [RG(window=w)]
        featurators[f"RGB-{w}"] = [RGB(window=w)]
        featurators[f"Gray-{w}"] = [Gray(window=w)]

    for w, d in ((11, 1), (11, 3), (21, 1), (21, 3), (31, 3), (51, 5), (51, 12)):
        featurators[f"Fourier-W{w}-D{d}"] = [Fourier(window=w, downsample=d)]

    for w1, w2, d in ((5, 11, 1), (5, 31, 3)):
        featurators[f"RGB-{w1}-Fourier-W{w2}-D{d}"] = [
            RGB(window=w1),
            Fourier(window=w2, downsample=d),
        ]
        featurators[f"RGB-{w1}-EfficientNet"] = [
            RGB(window=w1),
            EfficientNetB1(),
        ]

    for d in [1, 4, 8, 16]:
        for k in [64, 128, 512]:
            key = f"M-D{d}-{k}"
            featurators[key] = [Mosaiks(mosdir.joinpath(key))]

    # NOTE: featurators is a dictionary, so we can't rely on the *order* of
    # windows matching any particular order. Always use lookups
    windows = set([f.window for fset in featurators.values() for f in fset])

    return featurators, windows


def extract_patches(imdir, downsample, windows, names, sample_sets):
    return {
        name: get_patches(
            imdir=imdir,
            samples=samples,
            downsample=downsample,
            windows=windows,
        )
        for name, samples in zip(names, sample_sets)
    }


def extract_vectors(patches, fset, names):
    return {
        name: numpy.hstack(
            [
                featurator.transform(patches[name][featurator.window])
                for featurator in fset
            ]
        )
        for name in names
    }


def train_classifiers(
    imdir,
    train_samples,
    mosdir,
    pca_vis=False,
    val_size=0.25,
    test_size=0.15,
):

    featurators, windows = get_featurators(mosdir)

    classifiers = {
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(n_estimators=50),
        "RBF-SVM": SVC(kernel="rbf"),
        # NOTE: Too slow to run regularly, AND was consistently a low to mid
        # performer
        # "Linear SVM": SVC(kernel="linear"),
    }

    downsamplings = [1, 2, 4, 8]

    # Initialize tqdm with the total number of iterations
    progress_bar = tqdm(
        total=len(downsamplings) * len(featurators) * len(classifiers),
        desc="Training",
    )

    saved_classifiers = defaultdict(list)

    for downsample in downsamplings:

        # Get all patches at once to save time re-opening all those images
        patches = extract_patches(
            imdir, downsample, windows, ["train"], [train_samples]
        )

        for fname, fset in featurators.items():

            vectors = extract_vectors(patches, fset, ["train"])

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

                # Train
                classifier.fit(train_vectors, train_labels)

                # Save the trained classifier
                path = f"/tmp/{fname}:{cname}:{downsample}.pkl"
                joblib.dump(classifier, path)
                saved_classifiers[downsample].append(path)

                # Update the progress bar by one step
                progress_bar.update(1)

    progress_bar.close()

    return saved_classifiers


def supervised_test(
    imdir, mosdir, val_samples, test_samples, saved_classifiers, patch_images=False
):

    featurators, windows = get_featurators(mosdir)

    val_results = {}
    test_results = {}

    # Initialize tqdm with the total number of iterations
    progress_bar = tqdm(
        total=sum([len(v) for v in saved_classifiers.values()]),
        desc="Testing",
    )

    for downsample, classifier_sets in saved_classifiers.items():

        # Get all patches at once to save time re-opening all those images
        patches = extract_patches(
            imdir, downsample, windows, ["val", "test"], [val_samples, test_samples]
        )

        for classifier_path in classifier_sets:

            # Figure out which featurator we're referencing
            cname = Path(classifier_path).stem
            fname = cname.split(":")[0]
            fset = featurators[fname]

            # Load
            vectors = extract_vectors(patches, fset, ["val", "test"])
            classifier = joblib.load(classifier_path)

            # And test
            for samples, key, results in (
                (test_samples, "test", test_results),
                (val_samples, "val", val_results),
            ):
                actual = get_labels(samples)
                predicted = classifier.predict(vectors[key])
                results[cname] = {
                    "accuracy": accuracy_score(actual, predicted),
                    "F1": f1_score(actual, predicted, average="weighted"),
                    "actual": actual,
                    "predicted": predicted.tolist(),
                }

            if patch_images:
                max_window = max([f.window for f in fset])
                random_patch_vis(
                    patches=patches["val"][max_window],
                    actual=actual,
                    predicted=predicted,
                    number=(10, 16),
                    savedir=Path("/tmp"),
                    name=cname,
                )

            # Update the progress bar by one step
            progress_bar.update(1)

    return val_results, test_results


def report_results(results, only_rank=False, only=None):

    if not only_rank:
        for cname, stats in sorted(results.items()):
            print(
                f"{cname:<40} accuracy: {stats['accuracy']:.3f}, F1: {stats['F1']:.3f}"
            )
            save_confusion(cname, stats)

    if only is None:
        rank_stat(results, "accuracy", suffix="_val")
        best = rank_stat(results, "accuracy", top_x=15, suffix="_val")
        rank_stat(results, "accuracy", bottom_x=25, suffix="_val")
        return best

    else:
        # Only display these names
        results = {key: results[key] for key in only}
        rank_stat(results, "accuracy", suffix="_test")
        return None


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
        "--mosaik-dir",
        help="Directory where a MOSAIKS featurator can be found",
        type=Path,
    )
    parser.add_argument(
        "-o",
        "--only-rank",
        help="If True, only display ranked results, not individual",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--pca-vis",
        help="Whether to run vis of each vector set into /tmp/. Somewhat slow",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--saved-results",
        help="json file with results we want to re-view",
        type=Path,
    )
    parser.add_argument(
        "-t",
        "--patch-images",
        help="Whether to run vis of patches into /tmp/. Somewhat slow",
        action="store_true",
    )
    args = parser.parse_args()

    if args.saved_results:
        val_results = json.load(args.saved_results.open("r"))
    else:
        assert args.imdir.is_dir(), f"{args.imdir.absolute()} is not a directory"
        assert args.samples.is_file(), f"{args.samples.absolute()} is not a file"

        profile = Profile()
        profile.enable()

        # Ingest samples, throwing away random samples until the class numbers are
        # evenly split
        train_samples, val_samples, test_samples = split_by_image(
            ingest(args.samples, make_even=True),
            val_size=0.25,
            test_size=0.15,
        )

        saved_classifiers = train_classifiers(
            imdir=args.imdir,
            train_samples=train_samples,
            mosdir=args.mosaik_dir,
            pca_vis=args.pca_vis,
        )

        val_results, test_results = supervised_test(
            imdir=args.imdir,
            mosdir=args.mosaik_dir,
            val_samples=val_samples,
            test_samples=test_samples,
            saved_classifiers=saved_classifiers,
            patch_images=args.patch_images,
        )

        for key, results in (("val", val_results), ("test", test_results)):
            path = Path(f"/tmp/sweep_{key}_results_{int(time.time())}.json")
            json.dump(results, path.open("w"), indent=4, sort_keys=True)
            print(f"Results saved to {path}")

        profile.disable()
        path = "/tmp/compare.snakeviz"
        profile.dump_stats(path)
        print(f"Profile data saved to {path}")

    best = report_results(val_results, only_rank=args.only_rank)
    report_results(test_results, only_rank=args.only_rank, only=best)

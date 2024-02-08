"""
File for high-level tests of the lower-level components
"""

import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from mosaiks import Mosaiks
from sampling import get_patches, get_labels, ingest, smart_downsample


def supervised_test(imdir, sample_path, mosdir, test_size=0.3):

    samples = ingest(sample_path)

    train_samples, test_samples = train_test_split(
        samples,
        test_size=test_size,
        random_state=42,
    )

    featurators = {
        "MOSAIKS": Mosaiks(mosdir),
        # "TODO": TODO(savedir),
        # "TODO": TODO(savedir),
    }

    # NOTE: Linear SVMs take a long time to fit on many data points. Downsample
    # the points for them somehow.
    classifiers = {
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=50),
        "RBF SVM": SVC(kernel="rbf"),
        "Linear SVM": SVC(kernel="linear"),
    }
    # TODO: Expand these limits later (slowing runtime) for performance
    classifier_limits = {
        "Linear SVM": 200,
    }

    results = {}

    for downsample in [1, 2, 4, 8]:

        for fname, featurator in featurators.items():

            vectors = {}
            for name, samples in [("train", train_samples), ("test", test_samples)]:
                patches = get_patches(
                    imdir=imdir,
                    samples=samples,
                    downsample=downsample,
                    window=featurator.window,
                )
                vectors[name] = featurator.transform(patches)
            print("Vectors calculated")

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
                }

    for cname, stats in sorted(results.items()):
        print(f"{cname:<30} accuracy: {stats['accuracy']:.3f}, F1: {stats['F1']:.3f}")


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
        help="Path to collected samples",
        type=Path,
    )
    parser.add_argument(
        "-m",
        "--mosaic-dir",
        help="Directory where a MOSAIKS featurator can be found",
        type=Path,
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
    )

    profile.disable()
    profile.dump_stats("/tmp/compare.snakeviz")

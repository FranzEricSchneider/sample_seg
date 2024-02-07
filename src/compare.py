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

from sampling import get_patches, ingest


def supervised_test(sample_path, mosdir, test_size=0.3):

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

    classifiers = {
        "Linear SVM": SVC(kernel="linear"),
        "RBF SVM": SVC(kernel="rbf"),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=50),
    }

    results = {}

    for downsample in [1, 2, 4, 8]:
        for fname, featurator in featurators.items():
            for cname, classifier in classifiers.items():

                vectors = {}
                for name, samples in [("train", train_samples), ("test", test_samples)]:

                    pass
                    # patches = get_patches(
                    #     samples,
                    #     downsample=downsample,
                    #     window=featurator.window,
                    # )
                    # vectors[name] = featurator.transform(patches)

                # Train
                classifier.fit(vectors["train"], get_labels(train_samples))

                # And test
                actual = get_labels(test_samples)
                predicted = classifier.predict(vectors["test"])
                results[f"{fname}/{cname}/{downsample}"] = {
                    "accuracy": accuracy_score(actual, predicted),
                    "F1": f1_score(actual, predicted, average="weighted"),
                }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
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

    assert args.samples.is_file(), f"{args.samples.absolute()} is not a file"

    supervised_test(args.samples)

import cv2
from matplotlib import pyplot
import numpy
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import time
from tqdm import tqdm

from colors import RGB
from fourier import Fourier
from mosaiks import Mosaiks
from nn import EfficientNetB1
from sampling import get_patches


def timeit(
    fname, featurator, cname, classifier, impath, start=100, end=3000, number=20
):

    image = cv2.imread(str(impath))
    results = {}

    sizes = list(map(int, numpy.linspace(start, end, number)))

    for size in tqdm(sizes):

        results[size] = {}

        ipix = numpy.random.randint(0, image.shape[0], size=size)
        jpix = numpy.random.randint(0, image.shape[1], size=size)

        samples = [[impath.name, [i, j], "unknown"] for i, j in zip(ipix, jpix)]

        t1 = time.time()
        patches = get_patches(
            imdir=impath.parent,
            samples=samples,
            downsample=4,
            windows=[featurator.window],
        )
        t2 = time.time()
        results[size]["1) get_patches"] = t2 - t1

        t1 = time.time()
        vectors = featurator.transform(patches[featurator.window])
        t2 = time.time()
        results[size]["2) transform"] = t2 - t1

        # Fit the classifier to some nonsense just to get the computation. Just
        # make sure this doesn't count towards the time
        if size == start:
            classifier.fit(vectors[:100], [int(i / 51) for i in range(100)])

        t1 = time.time()
        classifier.predict(vectors)
        t2 = time.time()
        results[size]["3) predict"] = t2 - t1

    figure = pyplot.figure(figsize=(4, 3))
    pyplot.xscale("log")
    pyplot.yscale("log")

    x = sizes
    for name, keys, color in (
        ("patches", ["1) get_patches"], "g"),
        ("+transform", ["1) get_patches", "2) transform"], "k"),
        ("+predict", ["1) get_patches", "2) transform", "3) predict"], "r"),
    ):
        y = [sum([results[size][key] for key in keys]) for size in x]
        pyplot.plot(x, y, f"{color}o-", label=name)

    pyplot.ylim(0.05, 10)
    pyplot.xlim(100, 100000)
    pyplot.xlabel("Number of points sampled")
    pyplot.ylabel("Time (s)")
    pyplot.title(f"{fname} and {cname}")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.savefig(f"/tmp/sizing_{fname}_{cname}.png")


if __name__ == "__main__":

    for fname, featurator, p1, p2 in (
        ["RGB-5", RGB(window=5), 1000, 100000],
        ["Fourier-W31-D3", Fourier(window=31, downsample=3), 1000, 100000],
        [
            "M-D4-128",
            Mosaiks(Path("/home/fschneider/Downloads/MOSAIKS/M-D4-128")),
            1000,
            100000,
        ],
        ["EfficientNet", EfficientNetB1(), 100, 5000],
    ):
        for cname, classifier in (
            ["KNN", KNeighborsClassifier(n_neighbors=5)],
            ["Decision-Tree", DecisionTreeClassifier()],
            ["Random-Forest", RandomForestClassifier(n_estimators=50)],
            ["RBF-SVM", SVC(kernel="rbf")],
        ):
            timeit(
                fname=fname,
                featurator=featurator,
                cname=cname,
                classifier=classifier,
                impath=Path(
                    "/home/fschneider/Downloads/LABELLING/DOWNLOAD/1695400914000000_10_100_2_108.jpg"
                ),
                start=p1,
                end=p2,
            )

import cv2
from matplotlib import pyplot
import numpy
from sklearn.decomposition import PCA


# RGB colors
NONPLANT = [15, 15, 15]
PLANT = [0, 145, 25]
CORRECT = [0, 255, 0]
INCORRECT = [255, 0, 0]


def ij_slice(i, j, square, window):
    ipix = i * square + (square - window) // 2
    jpix = j * square + (square - window) // 2
    return slice(ipix, ipix + window), slice(jpix, jpix + window)


def window_slices(i, j, square, window):
    # Define the four regions we need
    ipix = [
        i * square + 1,
        i * square + (square - window) // 2 - 1,
        i * square + (square + window) // 2 + 1,
        (i + 1) * square - 1,
    ]
    jpix = [
        j * square + 1,
        j * square + (square - window) // 2 - 1,
        j * square + (square + window) // 2 + 1,
        (j + 1) * square - 1,
    ]
    return [
        (slice(ipix[0], ipix[3]), slice(jpix[0], jpix[1])),
        (slice(ipix[0], ipix[3]), slice(jpix[2], jpix[3])),
        (slice(ipix[0], ipix[1]), slice(jpix[0], jpix[3])),
        (slice(ipix[2], ipix[3]), slice(jpix[0], jpix[3])),
    ]


def half_slices(i, j, square, window):
    # Define the three regions we need
    ipix = [
        i * square + square // 2,
        i * square + (square + window) // 2 + 1,
        (i + 1) * square - 1,
    ]
    jpix = [
        j * square + 1,
        j * square + (square - window) // 2 - 1,
        j * square + (square + window) // 2 + 1,
        (j + 1) * square - 1,
    ]
    return [
        (slice(ipix[0], ipix[2]), slice(jpix[0], jpix[1])),
        (slice(ipix[0], ipix[2]), slice(jpix[2], jpix[3])),
        (slice(ipix[1], ipix[2]), slice(jpix[0], jpix[3])),
    ]


def write_rgb(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def random_patch_vis(patches, actual, predicted, number, savedir, name):

    chosen = numpy.random.choice(
        range(len(patches)), size=numpy.prod(number), replace=False
    )
    patches = [patches[i] for i in chosen]
    actual = [actual[i] for i in chosen]
    predicted = [predicted[i] for i in chosen]

    window = patches[0].shape[0]
    square = window + 10

    image = numpy.zeros((square * number[0], square * number[1], 3), dtype=numpy.uint8)

    index = 0
    for i in range(number[0]):
        for j in range(number[1]):
            islice, jslice = ij_slice(i, j, square, window)
            image[islice, jslice] = patches[index]
            index += 1

    # Write the raw patches as an image
    write_rgb(savedir.joinpath(f"patch_vis_raw_{name}.png"), image)

    index = 0
    for i in range(number[0]):
        for j in range(number[1]):
            color = PLANT if actual[index] == 1 else NONPLANT
            for islice, jslice in window_slices(i, j, square, window):
                image[islice, jslice] = color
            index += 1

    # Then color in whether the patch should be a plant or not
    write_rgb(savedir.joinpath(f"patch_vis_labeled_{name}.png"), image)

    index = 0
    for i in range(number[0]):
        for j in range(number[1]):
            color = CORRECT if actual[index] == predicted[index] else INCORRECT
            for islice, jslice in half_slices(i, j, square, window):
                image[islice, jslice] = color
            index += 1

    # Then color in whether the patch was classified correctly
    write_rgb(savedir.joinpath(f"patch_vis_classified_{name}.png"), image)


def vector_vis(vectors, labels, savedir, name):

    figure = pyplot.figure(figsize=(9, 9))

    # Check for really low-dim stuff (like greyscale)
    pca = None
    if vectors.shape[1] == 1:
        vectors2d = numpy.hstack([vectors] * 2)
    elif vectors.shape[1] == 2:
        vectors2d = vectors
    else:
        # Apply PCA to compress vectors down to 2D
        pca = PCA(n_components=2)
        vectors2d = pca.fit_transform(vectors)

    labels = numpy.array(labels)
    v0 = vectors2d[labels == 0]
    v1 = vectors2d[labels == 1]

    for v, c, alpha, label in (
        (v0, "black", 1.0, "Not Plant"),
        (v1, "forestgreen", 0.6, "Plant"),
    ):
        pyplot.scatter(v[:, 0], v[:, 1], color=c, label=label, alpha=alpha)

    # Include explained variance ratio in x and y labels
    if pca is None:
        pyplot.xlabel("Straight vector representation (low dimensionality)")
        pyplot.ylabel("Straight vector representation (low dimensionality)")
    else:
        evr = pca.explained_variance_ratio_
        pyplot.xlabel(f"Principal Component 1 ({evr[0]*100:.2f}% Variance)")
        pyplot.ylabel(f"Principal Component 2 ({evr[1]*100:.2f}% Variance)")

    pyplot.title(f"{name} Vector Visualization")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.savefig(savedir.joinpath(f"vector2d_vis_{name}.png"))
    pyplot.close(figure)

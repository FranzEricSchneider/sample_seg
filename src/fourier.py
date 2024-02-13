"""
https://www.smbc-comics.com/comics/20130201.gif
"""

import numpy


class Fourier:
    def __init__(self, window, downsample):
        self.window = window
        self.downsample = downsample
        assert isinstance(downsample, int)
        assert downsample > 0

    def transform(self, patches):

        vectors = []
        for patch in patches:
            vectors.append(
                numpy.hstack(
                    [
                        numpy.abs(numpy.fft.fft2(patch[:, :, channel]))[
                            :: self.downsample, :: self.downsample
                        ].flatten()
                        for channel in range(3)
                    ]
                )
            )
        return numpy.array(vectors)

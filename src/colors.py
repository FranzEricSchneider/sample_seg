import cv2

from mosaiks import flatten_patches


class RG:
    def __init__(self, window):
        self.window = window

    def transform(self, patches):
        rg = [patch[:, :, :2] for patch in patches]
        return flatten_patches(rg).T


class RGB:
    def __init__(self, window):
        self.window = window

    def transform(self, patches):
        return flatten_patches(patches).T


class Gray:
    def __init__(self, window):
        self.window = window

    def transform(self, patches):
        grayscale = [cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY) for patch in patches]
        return flatten_patches(grayscale).T

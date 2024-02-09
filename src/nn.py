from pathlib import Path

import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

from sampling import get_patches, get_labels, ingest, smart_downsample


class ResNet50:
    def __init__(self, snip=None):

        raw = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Drop the average pool and classifier layers no matter what
        modules = [_ for _ in raw.children()][:-2]

        if snip is None:
            self._model = torch.nn.Sequential(*modules)
        else:
            self._model = torch.nn.Sequential(*modules[:snip])
        self._model.eval()

        # Determined by hand, will change if the snip level changes (we should
        # do something smart and look up windows based on snip values later)
        # https://docs.google.com/presentation/d/1FK9NTFIgVcZR3eJtErEcl0lBm5vQWOHxEYgOMWb5gw8/edit#slide=id.g1f0c1002f0b_0_18
        self.window = 31

    def transform(self, patches):
        batch = load_images(patches)
        with torch.no_grad():
            vectors = torch.squeeze(self._model(batch))
        return vectors.numpy()


class EfficientNetB1:
    def __init__(self, snip=-2):

        raw = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)

        # Drop the average pool and classifier layers no matter what
        modules = [_ for _ in raw.children()][0]

        if snip is None:
            self._model = torch.nn.Sequential(*modules)
        else:
            self._model = torch.nn.Sequential(*modules[:snip])
        self._model.eval()

        # Determined by hand, will change if the snip level changes (we should
        # do something smart and look up windows based on snip values later)
        # https://docs.google.com/presentation/d/1FK9NTFIgVcZR3eJtErEcl0lBm5vQWOHxEYgOMWb5gw8/edit#slide=id.g1f0c1002f0b_0_18
        self.window = 31

    def transform(self, patches):
        batch = load_images(patches)
        with torch.no_grad():
            vectors = torch.squeeze(self._model(batch))
        return vectors.numpy()


def load_images(patches):
    # Load and preprocess the list of images
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    # NOTE: transforms.ToTensor converts the images from 0-255 down to 0-1
    return torch.stack([preprocess(patch) for patch in patches])


if __name__ == "__main__":

    sample_path = Path("/home/fschneider/Downloads/seg_samples_v1.json")
    imdir = Path("/home/fschneider/Downloads/LABELLING/DOWNLOAD/")
    window = 31

    samples = ingest(sample_path)
    patches = get_patches(
        imdir=imdir,
        samples=samples,
        downsample=8,
        window=window,
    )

    batch = load_images(patches)

    mres = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    meff = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
    mres.eval()
    meff.eval()

    # Drop the average pooling / classifier right away
    res_modules = [_ for _ in mres.children()][:-2]
    eff_modules = [_ for _ in meff.children()][0]

    res_hack = torch.nn.Sequential(*res_modules)
    eff_hack = torch.nn.Sequential(*eff_modules[:-2])

    with torch.no_grad():
        ores = torch.squeeze(res_hack(batch))
        oeff = torch.squeeze(eff_hack(batch))

    print(f"resnet: {ores.numpy().shape}")
    print(f"efficientnet: {oeff.numpy().shape}")
    import ipdb

    ipdb.set_trace()
    pass

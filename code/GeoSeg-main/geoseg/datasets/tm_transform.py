
# https://pytorch.org/vision/0.16/_modules/torchvision/transforms/transforms.html#RandomHorizontalFlip
import torch
from torchvision.transforms import functional as F
from torch import Tensor
from typing import Tuple, Dict

def custom_vflip_image(inputs):
    if len(inputs.shape) == 4:
        return inputs[:, :, ::-1, :]
    elif len(inputs.shape == 3):
        return inputs[:, ::-1, :]
    
def custom_hflip_image(inputs):
    if len(inputs.shape) == 4:
        return inputs[:, :, :, ::-1]
    elif len(inputs.shape) == 3:
        return inputs[:, :, ::-1]


class RandomCropResize(torch.nn.Module):
    """
    Randomly crop and resize the given image with a given probability.
    If the image is torch Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.

    Args:
        crop_size (Tuple[int, int]): Size of the crop (height, width) for S2 images
        planet_ratio (int): Ratio of Planet image size to S2 image size
        p (float): Probability of applying the transform. Default value is 1.0
    """

    def __init__(self, crop_size: Tuple[int, int], planet_ratio: int = 3, p: float = 1.0):
        super().__init__()
        self.crop_size = crop_size
        self.planet_ratio = planet_ratio
        self.p = p

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_dict (Dict[str, torch.Tensor]): Dictionary containing input tensors

        Returns:
            Dict[str, torch.Tensor]: Dictionary with randomly cropped and resized tensors
        """
        if torch.rand(1) < self.p:
            # Get the shape of the labels
            label_shape = input_dict["labels"].shape
            h, w = label_shape
            
            # Calculate the crop position for S2
            top = torch.randint(0, h - self.crop_size[0] + 1, (1,)).item()
            left = torch.randint(0, w - self.crop_size[1] + 1, (1,)).item()

            for modality in input_dict["non_empty_modalities"]:
                if modality == "planet":
                    # Adjust crop position and size for Planet
                    planet_top = top * self.planet_ratio
                    planet_left = left * self.planet_ratio
                    planet_crop_size = (self.crop_size[0] * self.planet_ratio, self.crop_size[1] * self.planet_ratio)
                    
                    # Crop Planet image
                    cropped = input_dict[modality][..., planet_top:planet_top+planet_crop_size[0], planet_left:planet_left+planet_crop_size[1]]
                    
                    # Resize back to original Planet size
                    input_dict[modality] = F.resize(cropped, (h * self.planet_ratio, w * self.planet_ratio), interpolation=F.InterpolationMode.BILINEAR)
                else:
                    # Crop other modalities
                    cropped = input_dict[modality][..., top:top+self.crop_size[0], left:left+self.crop_size[1]]
                    
                    # Resize back to original size
                    input_dict[modality] = F.resize(cropped, (h, w), interpolation=F.InterpolationMode.BILINEAR)
            
            # Handle labels separately
            if "labels" in input_dict:
                cropped_labels = input_dict["labels"][top:top+self.crop_size[0], left:left+self.crop_size[1]]
                input_dict["labels"] = F.resize(cropped_labels.unsqueeze(0), (h, w), interpolation=F.InterpolationMode.NEAREST).squeeze(0)

        return input_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(crop_size={self.crop_size}, planet_ratio={self.planet_ratio}, p={self.p})"



    
class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input_dict):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        apply_flip = torch.rand(1) < self.p
        if apply_flip:
            for modality in input_dict["non_empty_modalities"]:
                input_dict[modality] = custom_hflip_image(input_dict[modality])
                # if modality == "planet":
                #     input_dict[f"{modality}_masks"] = custom_hflip_image(input_dict[f"{modality}_masks"])
            input_dict["labels"] = input_dict["labels"][:, ::-1]
        return input_dict


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomVerticalFlip(torch.nn.Module):
    """Vertically flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input_dict):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        apply_flip = torch.rand(1) < self.p
        if apply_flip:
            for modality in input_dict["non_empty_modalities"]:
                input_dict[modality] = custom_vflip_image(input_dict[modality])
                # if modality == "planet":
                #     input_dict[f"{modality}_masks"] = custom_vflip_image(input_dict[f"{modality}_masks"])
            input_dict["labels"] = input_dict["labels"][::-1, :]
        return input_dict


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

class ToTensor:
    """Convert a PIL Image or ndarray to tensor and scale the values accordingly.

    This transform does not support torchscript.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.

    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks. See the `references`_ for implementing the transforms for image masks.

    .. _references: https://github.com/pytorch/vision/tree/main/references/segmentation
    """

    def __init__(self) -> None:
        pass

    def __call__(self, input_dict):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        for modality in input_dict["non_empty_modalities"]:
            input_dict[modality] = torch.tensor(input_dict[modality].copy()).to(dtype=torch.float32)
            # if modality == "planet":
            #     input_dict[f"{modality}_masks"] = torch.tensor(input_dict[f"{modality}_masks"].copy()).to(dtype=torch.float32)
        input_dict["labels"] = torch.tensor(input_dict["labels"].copy()).to(dtype=torch.int64)
        return input_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
class Normalize(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, modality_config, inplace=False):
        super().__init__()
        self.modality_config = modality_config
        self.inplace = inplace

    def forward(self, input_dict):
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        for modality in input_dict["non_empty_modalities"]:
            input_dict[modality] = F.normalize(input_dict[modality], self.modality_config[modality]["mean"], self.modality_config[modality]["std"], self.inplace)
        return input_dict


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.modality_config["s2"]["mean"]}, std={self.modality_config["s2"]["std"]})"
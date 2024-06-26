from PIL import Image
import blobfile as bf
# from mpi4py import MPI
import numpy as np
import random
import torch
import os
from torch.utils.data import DataLoader, Dataset


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, config={}
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    if "lizard" in data_dir:
        all_files = _list_image_files_recursively(os.path.join(data_dir, "classes"))
        dataset = NucleiMaskDataset(
            mask_paths=all_files, 
            resolution=(image_size, image_size),
            num_classes=7, 
            is_train=True,
            shard=config.rank,
            num_shards=config.world_size,
        )
    else:
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        if class_cond:
            # Assume classes are the first part of the filename,
            # before an underscore.
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            shard=config.rank,
            num_shards=config.world_size,
        )
    # Number of workers is set to 0 for debugging otherwise 1
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def resize_arr(pil_class, image_size, keep_aspect=True):
    pil_class = pil_class.resize(image_size, resample=Image.NEAREST)
    arr_class = np.array(pil_class)
    return arr_class


class NucleiMaskDataset(Dataset):
    def __init__(
        self,
        mask_paths,
        resolution,
        num_classes=7,
        shard=0,
        num_shards=1,
        random_crop=True,
        random_flip=True,
        is_train=True,
    ):
        super().__init__()
        self.is_train = is_train
        self.resolution = resolution
        self.local_masks = mask_paths[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.num_classes = num_classes

    def __len__(self):
        return len(self.local_masks)

    def __getitem__(self, idx):
        out_dict = {}
        # Load mask
        path = self.local_masks[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_mask = Image.open(f)
            pil_mask.load()
        pil_mask = pil_mask.convert("L")

        # Resize mask
        arr_mask = resize_arr(pil_mask, self.resolution)

        # Random flip
        # if self.random_flip and random.random() < 0.5:
        #     arr_mask = arr_mask[:, ::-1].copy()

        arr_mask = arr_mask[None, None, ]

        # To tensor
        mask_tensor = torch.from_numpy(arr_mask).long()
        bs, _, ht, wt = mask_tensor.size()
        nc = self.num_classes
        input_mask = torch.FloatTensor(bs, nc, ht, wt).zero_()
        input_mask = input_mask.scatter_(1, mask_tensor, 1.0)

        # Get condition
        cond = torch.sum(torch.sum(input_mask, dim=2), dim=2).bool().int().float().squeeze(0)
        # Remove the bs dimension
        input_mask = input_mask.squeeze(0)
 
        out_dict["y"] = cond
        
        # Return Image
        return input_mask, out_dict


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict

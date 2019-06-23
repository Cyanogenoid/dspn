import os

import h5py
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


class CLEVR_Images(torch.utils.data.Dataset):
    """ Dataset for MSCOCO images located in a folder on the filesystem """

    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path
        self.id_to_filename = self._find_images()
        self.sorted_ids = sorted(
            self.id_to_filename.keys()
        )  # used for deterministic iteration order
        print("found {} images in {}".format(len(self), self.path))
        self.transform = transform

    def _find_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith(".png"):
                continue
            id_and_extension = filename.split("_")[-1]
            id = int(id_and_extension.split(".")[0])
            id_to_filename[id] = filename
        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)


def create_coco_loader(path):
    transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )
    dataset = CLEVR_Images(path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, num_workers=12, shuffle=False, pin_memory=True
    )
    return data_loader


def main():
    for split_name in ["train", "val"]:
        path = os.path.join("clevr", "images", split_name)
        loader = create_coco_loader(path)
        images_shape = (len(loader.dataset), 3, 128, 128)

        with h5py.File("{}-images.h5".format(split_name), libver="latest") as fd:
            images = fd.create_dataset("images", shape=images_shape, dtype="float32")
            image_ids = fd.create_dataset(
                "image_ids", shape=(len(loader.dataset),), dtype="int32"
            )

            i = 0
            for ids, imgs in tqdm(loader):
                j = i + imgs.size(0)
                images[i:j, :, :] = imgs.numpy()
                image_ids[i:j] = ids.numpy().astype("int32")
                i = j


if __name__ == "__main__":
    main()

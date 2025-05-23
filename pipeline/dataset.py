import PIL.Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from pipeline.helpers import CARD_VALS, CARD_SUITS
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import os

class CardDataset(Dataset):
    '''
    A custom Pytorch Dataset which reads each deck specified in
    the constructor. Will return 13 or 52-class labels as requested.
    '''
    def __init__(self, decks: list[str], num_classes: int, do_augment: bool) -> None:
        self.decks = decks
        self.num_classes = num_classes
        self.counts: dict[str, int] = {}
        self.img_src = self.__load_data(do_augment)

    def __load_data(self, do_augment: bool) -> dict[str, int]:
        '''
        Uses the class's decks property to load every image into
        a dictionary. Counts the number of entries for each
        type of card.
        '''
        data = {}
        transform = transforms.Compose([
            transforms.ToTensor(),
            v2.GaussianNoise(sigma=.02),
            transforms.GaussianBlur(7)
        ]) if do_augment else transforms.Compose([
            transforms.ToTensor(),
        ])

        # iterate through dataset and add to data
        for deck in self.decks:
            for val in CARD_VALS:
                for suit in CARD_SUITS:
                    # update counts of each card
                    card_names = os.listdir(f"processed/{deck}/{val}_of_{suit}")
                    if f"{val}_of_{suit}" not in self.counts:
                        self.counts[f"{val}_of_{suit}"] = len(card_names)
                    else:
                        self.counts[f"{val}_of_{suit}"] += len(card_names)

                    # add card to dataset
                    if f"{val}_of_{suit}" not in data:
                        data[f"{val}_of_{suit}"] = []
                    for card in card_names:
                        pil_image = PIL.Image.open(f"processed/{deck}/{val}_of_{suit}/{card}")
                        final_img = transform(pil_image)

                        # debug: show the image!
                        # plt.imshow(final_img[0], cmap='grey')
                        # plt.axis('off')
                        # plt.show()
                        # exit(1)

                        data[f"{val}_of_{suit}"].append(final_img)

        return data

    def __len__(self) -> int:
        '''
        Returns the number of cards in the dataset.
        '''
        ctr = 0
        for type in self.counts:
            ctr += self.counts[type]
        return ctr

    def __getitem__(self, idx) -> tuple[list[list[float]], int]:
        running_idx = idx
        idx_ctr = self.counts["ace_of_clubs"]
        type_ctr = 0

        while running_idx >= idx_ctr:
            running_idx -= self.counts[f"{CARD_VALS[type_ctr // 4]}_of_{CARD_SUITS[type_ctr % 4]}"]
            type_ctr += 1
            idx_ctr = self.counts[f"{CARD_VALS[type_ctr // 4]}_of_{CARD_SUITS[type_ctr % 4]}"]

        out_img = self.img_src[f"{CARD_VALS[type_ctr // 4]}_of_{CARD_SUITS[type_ctr % 4]}"][running_idx]
        out_label = type_ctr
        if self.num_classes != 52:
            out_label //= 4
        if self.num_classes == 10 and out_label > 9:
            out_label = 9

        return out_img, out_label
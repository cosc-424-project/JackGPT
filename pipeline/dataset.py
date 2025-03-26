import PIL.Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from pipeline.helpers import CARD_VALS, CARD_SUITS
import os

class CardDataset(Dataset):
    def __init__(self, decks: list[str], is_13=True):
        self.decks = decks
        self.is_13 = is_13
        self.counts: dict[str, int] = {}
        self.img_src = self.__load_data()

    def __load_data(self):
        data = {}
        transform = transforms.Compose([
            transforms.Grayscale(),
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
                        data[f"{val}_of_{suit}"].append(final_img)

        return data

    def __len__(self):
        ctr = 0
        for type in self.counts:
            ctr += self.counts[type]
        return ctr

    def __getitem__(self, idx):
        running_idx = idx
        idx_ctr = self.counts["ace_of_clubs"]
        type_ctr = 0

        while running_idx >= idx_ctr:
            running_idx -= self.counts[f"{CARD_VALS[type_ctr // 4]}_of_{CARD_SUITS[type_ctr % 4]}"]
            type_ctr += 1
            idx_ctr = self.counts[f"{CARD_VALS[type_ctr // 4]}_of_{CARD_SUITS[type_ctr % 4]}"]

        return self.img_src[f"{CARD_VALS[type_ctr // 4]}_of_{CARD_SUITS[type_ctr % 4]}"][running_idx], type_ctr // 4 if self.is_13 else type_ctr
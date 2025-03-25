import PIL.Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from pipeline.helpers import CARD_VALS, CARD_SUITS
import os

PICS_PER_CARD = 10

class CardDataset(Dataset):
    def __init__(self, is_13=True):
        self.is_13 = is_13
        self.counts: dict[str, int] = {}
        self.img_src = self.__load_data()

    def __load_data(self):
        data = []
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        # iterate through dataset and add to data
        print("Loading dataset...", end="", flush=True)
        ctr = 0
        deck_names = os.listdir("processed")
        for deck in deck_names:
            for val in CARD_VALS:
                for suit in CARD_SUITS:
                    card_names = os.listdir(f"processed/{deck}/{val}_of_{suit}")
                    if f"{val}_of_{suit}" not in self.counts:
                        self.counts[f"{val}_of_{suit}"] = len(card_names)
                    else:
                        self.counts[f"{val}_of_{suit}"] += len(card_names)
                    for card in card_names:
                        pil_image = PIL.Image.open(f"processed/{deck}/{val}_of_{suit}/{card}")
                        final_img = transform(pil_image)
                        data.append(final_img)
                        ctr += 1
        print("done")
        
        return data

    def __len__(self):
        return len(self.img_src)

    def __getitem__(self, idx):
        idx_ctr = self.counts["ace_of_clubs"]
        type_ctr = 0

        while idx > idx_ctr:
            type_ctr += 1
            idx_ctr += self.counts[f"{CARD_VALS[type_ctr // 4]}_of_{CARD_SUITS[type_ctr % 4]}"]

        return self.img_src[idx], type_ctr // 4 if self.is_13 else type_ctr
    

if __name__ == "__main__":
    cd13 = CardDataset()
    print(cd13.__getitem__(0)[0].shape)
    # 126x90
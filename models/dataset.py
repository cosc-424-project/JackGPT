import PIL.Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CardDataset(Dataset):
    def __init__(self, is_13=True):
        self.is_13 = is_13
        self.img_src = self.__load_data()

    def __load_data(self):
        vals = [
            "ace",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "jack",
            "queen",
            "king",
        ]

        suits = [
            "clubs",
            "diamonds",
            "hearts",
            "spades",
        ]

        data = []
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        # iterate through dataset and add to data
        print("Loading dataset...", end="", flush=True)
        ctr = 0
        for val in vals:
            for suit in suits:
                # print(f"Processing {val} of {suit}...", flush=True)
                for i in range(50):
                    # print(f"data/{val}_of_{suit}.mp4/card_{i:02d}.png")
                    # print(f"\t[{ctr}]:  {ctr // 200}={vals[ctr//200]}  {ctr // 50 % 4}={suits[ctr//50%4]}  {ctr % 50}")
                    pil_image = PIL.Image.open(f"processed/{val}_of_{suit}/card_{i:02d}.png")
                    final_img = transform(pil_image)
                    data.append(final_img)
                    ctr += 1
        print("done")
        
        return data

    def __len__(self):
        return len(self.img_src)

    def __getitem__(self, idx):
        return self.img_src[idx], idx // 200 if self.is_13 else idx // 50
    

if __name__ == "__main__":
    cd13 = CardDataset()
    print(cd13.__getitem__(0)[0].shape)
    # 126x90
import torch
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import PIL.Image

model: torch.nn.Module = torch.load("./ignore_data/models/test_deck1+epoch_3.pt", weights_only=False)
transform = transforms.Compose([
    transforms.ToTensor(),
    v2.GaussianNoise(sigma=.02),
    transforms.GaussianBlur(7)
])
pil_image = PIL.Image.open(f"processed/deck1/ace_of_clubs/card_00.png")
final_img: torch.Tensor = transform(pil_image)

model.eval()
with torch.inference_mode():
    print(model(final_img.unsqueeze(0)))
import cv2
import numpy as np
from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import PIL.Image

MIN_AREA = 2000
OUTPUT_WIDTH, OUTPUT_HEIGHT = 450, 635
CLASSES = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "Face/10"]


def image_preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, filtered = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    img_erosion = cv2.erode(filtered, kernel, iterations=1)

    return img_erosion


def get_contours(processed_image):
    canny_output = cv2.Canny(processed_image, 100, 200)
    contours, _ = cv2.findContours(
        canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return [contour for contour in contours if cv2.contourArea(contour) > MIN_AREA]


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]
    rect[3] = pts[np.argmax(s)]

    return rect


def extract_card(image, box):
    output_box = np.float32(
        [[0, 0], [OUTPUT_WIDTH, 0], [0, OUTPUT_HEIGHT], [OUTPUT_WIDTH, OUTPUT_HEIGHT]]
    )

    (tl, tr, bl, br) = box
    width = np.linalg.norm(tr - tl)
    height = np.linalg.norm(bl - tl)

    if width > height:
        box = np.array([bl, tl, br, tr], dtype="float32")

    perspective_matrix = cv2.getPerspectiveTransform(box, output_box)
    warped = cv2.warpPerspective(
        image, perspective_matrix, (OUTPUT_WIDTH, OUTPUT_HEIGHT)
    )

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (90, 126))

    return resized


def main():
    load_dotenv()

    camera_ip_address = os.getenv("CAMERA_IP_ADDRESS")
    camera_port = os.getenv("PORT")

    url = f"http://{camera_ip_address}:{camera_port}/video"

    cap = cv2.VideoCapture(url)

    model: torch.nn.Module = torch.load(
        "models/test_deck2+epoch_3.pt", weights_only=False
    )
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            v2.GaussianNoise(sigma=0.02),
            transforms.GaussianBlur(7),
        ]
    )

    while True:
        ret, image = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        processed_image = image_preprocessing(image)
        contours = get_contours(processed_image)

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = order_points(box)

            card = extract_card(image, box)
            card = transform(card)

            with torch.inference_mode():
                logits = model(card.unsqueeze(0))
                probs = nn.Softmax(dim=1)(logits)
                predicted_idx = torch.argmax(probs, dim=1).item()

            # Font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.75
            color = (0, 255, 255)
            thickness = 2

            center = int(np.mean(box[:, 0])) - 10, int(np.mean(box[:, 1]))

            cv2.putText(
                image,
                CLASSES[predicted_idx],
                center,
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        cv2.imshow("IP Camera Feed", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

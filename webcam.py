import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import PIL.Image
import argparse

MIN_AREA = 2000
OUTPUT_WIDTH, OUTPUT_HEIGHT = 450, 635
CLASSES = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "Face/10"]
MODEL_PATHS = [
    os.path.join("models", "test_deck1+epoch_3.pt"),
    os.path.join("models", "test_deck2+epoch_3.pt"),
    os.path.join("models", "test_deck4+epoch_3.pt"),
]
TRANSFORMATIONS = transforms.Compose(
    [
        transforms.ToTensor(),
        v2.GaussianNoise(sigma=0.02),
        transforms.GaussianBlur(7),
    ]
)
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (0, 255, 255),
]


def image_preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, filtered = cv2.threshold(blurred, 175, 255, cv2.THRESH_BINARY_INV)

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


def extract_card(image, contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = order_points(box)

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

    transformed = TRANSFORMATIONS(resized)

    return transformed, box


def make_prediction(card, model):
    logits = model(card.unsqueeze(0))
    probs = nn.Softmax(dim=1)(logits)
    predicted_idx = torch.argmax(probs, dim=1).item()

    return CLASSES[predicted_idx]


def add_label(image, prediction, position, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2

    cv2.putText(
        image,
        prediction,
        np.intp(position),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def main():
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    models = [torch.load(path, weights_only=False).eval() for path in MODEL_PATHS]

    parser = argparse.ArgumentParser(
        prog="JackGPT Webcam",
        description="This script uses the webcam to classify playing cards",
    )

    parser.add_argument("--debug", action="store_true", help="Show debug options")

    args = parser.parse_args()

    while True:
        ret, image = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        processed_image = image_preprocessing(image)
        contours = get_contours(processed_image)

        for contour in contours:
            card, box = extract_card(image, contour)

            predictions = np.array([make_prediction(card, model) for model in models])

            if args.debug:
                for index, (prediction, color) in enumerate(zip(predictions, COLORS)):
                    add_label(image, prediction, box[index], color)

            values, counts = np.unique(predictions, return_counts=True)
            final_prediction = values[np.argmax(counts)]
            center = int(np.mean(box[:, 0])) - 10, int(np.mean(box[:, 1]))
            add_label(image, final_prediction, center, COLORS[-1])

        cv2.imshow("IP Camera Feed", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

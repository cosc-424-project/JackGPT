import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import argparse


MIN_AREA = 2000
OUTPUT_WIDTH, OUTPUT_HEIGHT = 450, 635
CLASSES = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "Face/10"]
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


# get trained models from filesystem
def get_model_path(num_of_classes):
    total_num_models = 5 if num_of_classes == 52 else 3

    return [
        os.path.join("models", f"class_{num_of_classes}_model_{num_of_models}.pt")
        for num_of_models in range(1, total_num_models + 1)
    ]


# get output class name from number of input classes and predictions
def get_class_name(num_of_classes, prediction_index):
    classes_10 = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "Face/10"]
    classes_13 = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    suits = ["C", "D", "H", "S"]

    if num_of_classes == 10:
        return classes_10[prediction_index]
    if num_of_classes == 13:
        return classes_13[prediction_index]
    if num_of_classes == 52:
        rank = classes_13[prediction_index // 4]
        suit = suits[prediction_index % 4]

        return f"{suit}{rank}"


# create binary frame representation
def image_preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, filtered = cv2.threshold(blurred, 175, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    img_erosion = cv2.erode(filtered, kernel, iterations=1)

    return img_erosion


# get contours from the image
def get_contours(processed_image):
    canny_output = cv2.Canny(processed_image, 100, 200)
    contours, _ = cv2.findContours(
        canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return [contour for contour in contours if cv2.contourArea(contour) > MIN_AREA]


# get corners of rectangles
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]
    rect[3] = pts[np.argmax(s)]

    return rect


# get card from image
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


# preform prediction on card
def make_prediction(card, model, num_of_classes):
    logits = model(card.unsqueeze(0))
    probs = nn.Softmax(dim=1)(logits)
    predicted_idx = torch.argmax(probs, dim=1).item()

    return get_class_name(num_of_classes, predicted_idx)


# add label to image
def add_label(image, prediction, position, color, font_scale=0.5):
    font = cv2.FONT_HERSHEY_SIMPLEX
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


def get_chosen_idx() -> int:
    # get the available video capture indices
    print("Searching for valid video devices... expect 1 error", flush=True)
    video_idxs = []
    cur_idx = 0
    found_invalid = False
    while not found_invalid:
        tmp_cap = cv2.VideoCapture(cur_idx)
        if not tmp_cap.isOpened():
            found_invalid = True
            break
        print(f"Found device #{cur_idx}", flush=True)
        tmp_cap.release()
        video_idxs.append(cur_idx)
        cur_idx += 1

    # if none available, error
    if len(video_idxs) == 0:
        print("Error: no available cameras.", flush=True)
        exit(1)

    # default if there's only one
    elif len(video_idxs) == 1:
        chosen_idx = video_idxs[0]
        print(f"Only available video device is #{chosen_idx}.", flush=True)

    # else, have user decide
    else:
        while chosen_idx == -1:
            try:
                chosen_idx = int(
                    input(f"Which camera would you like to use [0-{video_idxs[-1]}]? ")
                )
                if chosen_idx > video_idxs[-1] or chosen_idx < 0:
                    chosen_idx = -1
            except ValueError:
                continue
            except KeyboardInterrupt:
                exit()

    return chosen_idx


def main():
    chosen_idx = get_chosen_idx()
    cap = cv2.VideoCapture(chosen_idx)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # handle arguments
    parser = argparse.ArgumentParser(
        prog="JackGPT Webcam",
        description="This script uses the webcam to classify playing cards",
    )
    parser.add_argument("--debug", action="store_true", help="Show debug options")
    parser.add_argument(
        "--classes", type=int, help="The number of output classes", required=True
    )

    args = parser.parse_args()

    # load models
    models = [
        torch.load(path, weights_only=False).eval()
        for path in get_model_path(args.classes)
    ]

    while True:
        # frame from camera
        ret, image = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # get contours from frame
        processed_image = image_preprocessing(image)
        contours = get_contours(processed_image)

        # for each frame
        for contour in contours:
            # get the card
            card, box = extract_card(image, contour)

            # make prediction
            predictions = np.array(
                [make_prediction(card, model, args.classes) for model in models]
            )

            # add each model label
            if args.debug:
                for index, (prediction, color) in enumerate(zip(predictions, COLORS)):
                    add_label(image, prediction, box[index % 4], color)

            # add final label
            values, counts = np.unique(predictions, return_counts=True)
            final_prediction = values[np.argmax(counts)]
            center = int(np.mean(box[:, 0])) - 10, int(np.mean(box[:, 1]))
            add_label(image, final_prediction, center, COLORS[-1], font_scale=1)

        cv2.imshow("IP Camera Feed", cv2.resize(image, (1280, 960)))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

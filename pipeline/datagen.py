import cv2
import os
import numpy as np
import math
from pipeline.helpers import confirm, error, CARD_VALS, CARD_SUITS
from sys import stderr
from subprocess import run


OUTPUT_SIZE = (450, 635)
OUTPUT_IMAGE_DIR = "./images"
INPUT_VIDEO_DIR = "./videos"
NUM_OF_IMAGE_PER_CARD = 10


def get_rotmat(box: list[list[float]]):
    # get highest point in box and average point
    sum_x = 0
    sum_y = 0
    highest_idx = -1
    highest_y = None
    # ^^^^^^^ this is actually the lowest y, since top left is (0, 0)
    for i in range(4):
        sum_x += box[i][0]
        sum_y += box[i][1]
        if highest_y == None or box[i][1] < highest_y:
            highest_y = box[i][1]
            highest_idx = i

    center = (sum_x / 4, sum_y / 4)

    # get second furthest point
    distances = np.zeros((4))
    for i in range(4):
        distances[i] = 0 if i == highest_idx else math.sqrt(
            (box[highest_idx][0] - box[i][0]) * (box[highest_idx][0] - box[i][0]) +
            (box[highest_idx][1] - box[i][1]) * (box[highest_idx][1] - box[i][1])
        )

    distances[distances.argmax()] = 0
    p1 = box[highest_idx]
    p2 = box[distances.argmax()]

    # get angle between those two points
    angle_rad = math.atan(
        (p2[1] - p1[1]) / (p2[0] - p1[0] if abs(p2[0] - p1[0]) > .001 else .001)
    )
    angle_deg = angle_rad * 360 / (2 * math.pi)

    M = cv2.getRotationMatrix2D(center, 90 + angle_deg, 1)

    return M


def update_box(box: list[list[float]], rotmat: cv2.typing.MatLike):
    affine_mat = np.array([rotmat[0], rotmat[1], [0, 0, 1]])
    p1 = np.array([box[0][0], box[0][1], 1])
    p2 = np.array([box[1][0], box[1][1], 1])
    p3 = np.array([box[2][0], box[2][1], 1])
    p4 = np.array([box[3][0], box[3][1], 1])
    adjusted = np.array([p1, p2, p3, p4])
    updated = np.ndarray.astype((adjusted @ affine_mat.T)[:, :2], np.int32)
    return updated


def get_cropped(image, dev = False):
    # get gray scale of file
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # basic threshold and basic erosion
    MAX = gray.max()
    thresh = .15
    _ ,filtered = cv2.threshold(gray, MAX * (1 - thresh), 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(filtered, kernel, iterations=1) 

    # get contours of image
    canny_output = cv2.Canny(img_erosion, 100, 100 * 2)
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # extract bounding boxes of each contour
    minRect = [None]*len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv2.minAreaRect(c)
        
    # get largest bounding box
    max_box = None
    max_area = -1
    for rect in minRect:
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        area = 0.5 * abs(
            (box[0][0] * box[1][1] + box[1][0] * box[2][1] + box[2][0] * box[3][1] + box[3][0] * box[0][1]) -
            (box[0][1] * box[1][0] + box[1][1] * box[2][0] + box[2][1] * box[3][0] + box[3][1] * box[0][0])
        )
        if area > max_area:
            max_area = area
            max_box = box

    # OPTIONAL: draw bounding box on original image and show
    if dev:
        cv2.drawContours(image, [max_box], 0, (0, 0, 255))
        cv2.imshow('Cropped', image)
        cv2.waitKey()

    rotmat = get_rotmat(max_box)
    rotated = cv2.warpAffine(image, rotmat, (image.shape[1], image.shape[0]))

    updated = update_box(max_box, rotmat)
    max_x = updated[:, 0].max()
    min_x = updated[:, 0].min()
    max_y = updated[:, 1].max()
    min_y = updated[:, 1].min()
    cropped = rotated[min_y:max_y, min_x:max_x]
    return cropped


def process_image(image):
    cropped = get_cropped(image, False)
    resized = cv2.resize(cropped, (450, 635))
    return resized


def extract_frames(video_path, output_path, num_of_images=NUM_OF_IMAGE_PER_CARD):
    os.makedirs(output_path, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frames = np.sort(np.random.choice(number_of_frames, num_of_images, replace=False))
    
    frame_count = 0
    saved_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in random_frames:
            processed_frame = process_image(frame)
            frame_filename = os.path.join(output_path, f"card_{saved_frames:02d}.png")
            cv2.imwrite(frame_filename, processed_frame)
            saved_frames += 1

        frame_count += 1
    
    cap.release()


def create_images() -> None:
    os.makedirs("raw_images")
    deck_names = os.listdir("mp4_data")
    for deck_name in deck_names:
        os.makedirs(f"raw_images/{deck_name}")
        video_names = os.listdir(f"mp4_data/{deck_name}")
        for video in video_names:
            os.makedirs(f"raw_images/{deck_name}/{video.split('.')[0]}", exist_ok=True)
            extract_frames(
                f"mp4_data/{deck_name}/{video}",
                f"raw_images/{deck_name}/{video.split('.')[0]}"
            )


def to_mp4() -> None:
    os.makedirs("mp4_data")
    deck_names = os.listdir("data")
    for deck_name in deck_names:
        os.makedirs(f"mp4_data/{deck_name}", exist_ok=True)
        video_names = os.listdir(f"data/{deck_name}")
        for video in video_names:
            # ffmpeg -i input.mov -c:v copy -loglevel quiet -c:a copy output.mp4
            run([
                "ffmpeg",
                "-i",
                f"data/{deck_name}/{video}",
                "-c:v",
                "copy",
                "-loglevel",
                "quiet",
                "-c:a",
                "copy",
                f"mp4_data/{deck_name}/{video.split('.')[0] + '.mp4'}"
            ])


def preprocess():
    os.makedirs("processed", exist_ok=True)
    deck_names = os.listdir("raw_images")
    for deck_name in deck_names:
        os.makedirs(f"processed/{deck_name}", exist_ok=True)
        
        # iterate through every card name
        for val in CARD_VALS:
            for suit in CARD_SUITS:
                os.makedirs(f"processed/{deck_name}/{val}_of_{suit}", exist_ok=True)
                image_names = os.listdir(f"raw_images/{deck_name}/{val}_of_{suit}")
                for image in image_names:
                    orig = cv2.imread(f"raw_images/{deck_name}/{val}_of_{suit}/{image}")
                    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
                    resized_image = cv2.resize(gray, (90, 126))
                    cv2.imwrite(f"processed/{deck_name}/{val}_of_{suit}/{image}", resized_image)


def rename_mp4s():
    deck_names = os.listdir("mp4_data")
    for deck_name in deck_names:
        video_names = os.listdir(f"mp4_data/{deck_name}")
        ctr = 0
        for video in video_names:
            new_name = f"mp4_data/{deck_name}/{CARD_VALS[ctr % 13]}_of_{CARD_SUITS[ctr // 13]}.mp4"
            os.rename(f"mp4_data/{deck_name}/{video}", new_name)
            ctr += 1



def datagen() -> None:
    if not confirm("Is data already processed? "):
        print("Beginning dataset generation...")
        # does data dir exist?
        # are there more than two data subdirectories?
        if not os.path.isdir("data"):
            error("Error: data directory does not exist, read README for instructions.")
        data_subdirs = os.listdir("data")
        if len(data_subdirs) < 2:
            error("Error: two data subdirectories are required, read README for instructions.")
        
        # convert MOV videos into mp4 in mp4_data
        print("Converting MOV files to mp4 files...", end="", flush=True)
        to_mp4()
        print("done", flush=True)

        # rename mp4 files if necessary
        if not confirm("Are raw data files properly named? "):
            if confirm("Are your videos properly ordered for automatic renaming? "):
                print("Renaming mp4 files...", end="", flush=True)
                rename_mp4s()
                print("done", flush=True)
            else:
                print("Hit ENTER after renaming the files manually...")
                input()

        # convert mp4 files to images in raw_images
        print("Picking raw images from mp4 videos...", end="", flush=True)
        create_images()
        print("done", flush=True)

        # convert raw images to preprocessed images in processed
        print("Preprocessing images for card model...", end="", flush=True)
        preprocess()
        print("done", flush=True)
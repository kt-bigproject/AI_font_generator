# -*- coding: utf-8 -*-
import argparse
import sys
import glob
import numpy as np
import io, os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import collections

sys.path.append("../")
from utils.util import *

SRC_PATH = "../fonts/source/"
TRG_PATH = "../fonts/target/"
OUTPUT_PATH = "./dataset-11172/"
TEXT_PATH = "./2350-common-hangul.txt"
CANVAS_SIZE = 128


def draw_single_char(ch, font, canvas_size):
    image = Image.new("L", (canvas_size, canvas_size), color=255)
    drawing = ImageDraw.Draw(image)
    w, h = font.getbbox(ch)[2:4]
    drawing.text(
        ((canvas_size - w) / 2, (canvas_size - h) / 2), ch, fill=(0), font=font
    )
    flag = np.sum(np.array(image))

    # 해당 font에 글자가 없으면 return None
    if flag == 255 * 128 * 128:
        return None

    return image


def draw_example(ch, src_font, dst_font, canvas_size):
    dst_img = draw_single_char(ch, dst_font, canvas_size)

    # 해당 font에 글자가 없으면 return None
    if not dst_img:
        return None

    src_img = draw_single_char(ch, src_font, canvas_size)
    example_img = Image.new(
        "RGB", (canvas_size * 2, canvas_size), (255, 255, 255)
    ).convert("L")
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img


def draw_handwriting(ch, src_font, canvas_size, dst_folder, label, count):
    dst_path = dst_folder + "%d_%04d" % (label, count) + ".png"
    dst_img = Image.open(dst_path)
    src_img = draw_single_char(ch, src_font, canvas_size)
    example_img = Image.new(
        "RGB", (canvas_size * 2, canvas_size), (255, 255, 255)
    ).convert("L")
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img


def create_images_from_fonts():
    # Check if output directory exists, if not, create it
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Load the text file
    with open(TEXT_PATH, "r", encoding="utf-8") as file:
        chars = file.read().replace("\n", "")

    # Load the source font
    src_font_files = glob.glob(SRC_PATH + "*.*")
    assert len(src_font_files) > 0, "No source font file found"
    src_font = ImageFont.truetype(src_font_files[0], size=32)

    # For each target font
    for idx, trg_font_file in enumerate(glob.glob(TRG_PATH + "*.*")):
        trg_font = ImageFont.truetype(trg_font_file, size=32)

        # For each character
        for ch in chars:
            # Create image from source and target font
            example_img = draw_example(ch, src_font, trg_font, CANVAS_SIZE)

            # If the image is None, continue with the next character
            if example_img is None:
                continue

            # Convert PIL Image to numpy array
            example_np = np.array(example_img)

            # Split the concatenated image into two separate images
            img_A, img_B = read_split_image(example_np)

            # Crop, resize and pad each image to 128x128
            img_A = centering_image(
                img_A, image_size=128, verbose=False, resize_fix=True
            )
            img_B = centering_image(
                img_B, image_size=128, verbose=False, resize_fix=True
            )

            # Concatenate two images into a single image of size 128x256
            example_np = np.concatenate([img_A, img_B], axis=1)

            # Convert numpy array back to PIL Image
            example_img = Image.fromarray(example_np.astype(np.uint8))

            # Save the image
            image_file_name = OUTPUT_PATH + "%d_%s.png" % (idx, ch)
            example_img.save(image_file_name)


if __name__ == "__main__":
    create_images_from_fonts()

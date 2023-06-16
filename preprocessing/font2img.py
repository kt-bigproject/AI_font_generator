# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import argparse
import sys
import glob
import numpy as np
import io, os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from skimage.transform import resize
import collections

sys.path.append("../")
from utils import *

SRC_PATH = "../fonts/source/"
TRG_PATH = "../fonts/target/"
OUTPUT_PATH = "./dataset-2350/"
TEXT_PATH = "./2350-common-hangul.txt"
CANVAS_SIZE = 64


def draw_single_char(ch, font, canvas_size):
    image = Image.new("L", (canvas_size, canvas_size), color=255)
    drawing = ImageDraw.Draw(image)
    w, h = font.getbbox(ch)[2:4]
    drawing.text(
        ((canvas_size - w) / 2, (canvas_size - h) / 2), ch, fill=(0), font=font
    )
    flag = np.sum(np.array(image))

    # 해당 font에 글자가 없으면 return None
    if flag == 255 * CANVAS_SIZE * CANVAS_SIZE:
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


def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    # Load source font
    src_font = ImageFont.truetype(SRC_PATH + "source_font.ttf", CANVAS_SIZE - 8)

    # Load target fonts
    target_font_paths = glob.glob(TRG_PATH + "*.ttf")
    target_fonts = [
        ImageFont.truetype(font, CANVAS_SIZE - 8) for font in target_font_paths
    ]

    # Load text
    with io.open(TEXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    for idx, ch in enumerate(text):
        src_img = draw_single_char(ch, src_font, CANVAS_SIZE)
        if src_img is None:
            continue

        src_img = np.array(src_img)
        src_img = tight_crop_image(src_img)
        src_img = resize(src_img, (CANVAS_SIZE - 8, CANVAS_SIZE - 8))
        src_img = add_padding(src_img, image_size=CANVAS_SIZE)

        for font_idx, font in enumerate(target_fonts):
            trg_img = draw_single_char(ch, font, CANVAS_SIZE)
            if trg_img is None:
                continue

            trg_img = np.array(trg_img)
            trg_img = tight_crop_image(trg_img)
            trg_img = resize(trg_img, (CANVAS_SIZE - 8, CANVAS_SIZE - 8))
            trg_img = add_padding(trg_img, image_size=CANVAS_SIZE)

            final_img = np.concatenate([trg_img, src_img], axis=1)

            image = Image.new(
                "RGB", (CANVAS_SIZE * 2, CANVAS_SIZE), (255, 255, 255)
            ).convert("L")
            image.paste(Image.fromarray(final_img), (0, 0))
            final_img = final_img * 255
            final_img = final_img.astype(np.uint8)

            # Save the image
            output_path = os.path.join(
                OUTPUT_PATH, "{}_{}.png".format(idx, font_idx + 1)
            )
            Image.fromarray(final_img).save(output_path)


if __name__ == "__main__":
    main()

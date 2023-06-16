import os, glob, time, datetime
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.utils import save_image

from common.dataset import TrainDataProvider
from common.function import init_embedding
from common.models import Encoder, Decoder, Discriminator, Generator
from common.utils import denorm_image, centering_image


def interpolation(
    data_provider,
    grids,
    fixed_char_ids,
    interpolated_font_ids,
    embeddings,
    En,
    De,
    batch_size,
    img_size=128,
    save_nrow=6,
    save_path=False,
    GPU=True,
):
    train_batch_iter = data_provider.get_train_iter(batch_size, with_charid=True)

    for grid_idx, grid in enumerate(grids):
        train_batch_iter = data_provider.get_train_iter(batch_size, with_charid=True)
        grid_results = {
            from_to: {charid: None for charid in fixed_char_ids}
            for from_to in interpolated_font_ids
        }

        for i, batch in enumerate(train_batch_iter):
            font_ids_from, char_ids, batch_images = batch
            font_filter = [i[0] for i in interpolated_font_ids]
            font_filter_plus = font_filter + [font_filter[0]]
            font_ids_to = [
                font_filter_plus[font_filter.index(i) + 1] for i in font_ids_from
            ]
            batch_images = batch_images.cuda()

            real_sources = batch_images[:, 1, :, :].view(
                [batch_size, 1, img_size, img_size]
            )
            real_targets = batch_images[:, 0, :, :].view(
                [batch_size, 1, img_size, img_size]
            )

            for idx, (image_S, image_T) in enumerate(zip(real_sources, real_targets)):
                image_S = image_S.cpu().detach().numpy().reshape(img_size, img_size)
                image_S = centering_image(image_S, resize_fix=100)
                real_sources[idx] = torch.tensor(image_S).view([1, img_size, img_size])
                image_T = image_T.cpu().detach().numpy().reshape(img_size, img_size)
                image_T = centering_image(image_T, resize_fix=100)
                real_targets[idx] = torch.tensor(image_T).view([1, img_size, img_size])

            encoded_source, encode_layers = En(real_sources)

            interpolated_embeddings = []
            embedding_dim = embeddings.shape[3]
            for from_, to_ in zip(font_ids_from, font_ids_to):
                interpolated_embeddings.append(
                    (embeddings[from_] * (1 - grid) + embeddings[to_] * grid)
                    .cpu()
                    .numpy()
                )
            interpolated_embeddings = torch.tensor(interpolated_embeddings).cuda()
            interpolated_embeddings = interpolated_embeddings.reshape(
                batch_size, embedding_dim, 1, 1
            )

            # generate fake image with embedded source
            interpolated_embedded = torch.cat(
                (encoded_source, interpolated_embeddings), 1
            )
            fake_targets = De(interpolated_embedded, encode_layers)

            # [(0)real_S, (1)real_T, (2)fake_T]
            for fontid, charid, real_S, real_T, fake_T in zip(
                font_ids_from, char_ids, real_sources, real_targets, fake_targets
            ):
                font_from = fontid
                font_to = font_filter_plus[font_filter.index(fontid) + 1]
                from_to = (font_from, font_to)
                grid_results[from_to][charid] = [real_S, real_T, fake_T]

        if save_path:
            for from_to in grid_results.keys():
                image = [
                    grid_results[from_to][charid][2].cpu().detach().numpy()
                    for charid in fixed_char_ids
                ]
                image = torch.tensor(np.array(image))

                # path
                font_from = str(from_to[0])
                font_to = str(from_to[1])
                grid_idx = str(grid_idx)
                if len(font_from) == 1:
                    font_from = "0" + font_from
                if len(font_to) == 1:
                    font_to = "0" + font_to
                if len(grid_idx) == 1:
                    grid_idx = "0" + grid_idx
                idx = str(interpolated_font_ids.index(from_to))
                if len(idx) == 1:
                    idx = "0" + idx
                file_path = "%s_from_%s_to_%s_grid_%s.png" % (
                    idx,
                    font_from,
                    font_to,
                    grid_idx,
                )

                # save
                save_image(
                    denorm_image(image.data),
                    os.path.join(save_path, file_path),
                    nrow=save_nrow,
                    pad_value=255,
                )

    return grid_results

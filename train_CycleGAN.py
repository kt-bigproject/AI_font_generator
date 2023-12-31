import datetime
import glob
import os
import sys

sys.path.append("./")
import time

import numpy as np
import torch
import torch.nn as nn
import wandb
from data_loader.data_loader import TrainDataProvider
from model.function import init_embedding
from model.models import Decoder, Discriminator, Encoder, Generator
from PIL import Image
from torchvision.utils import save_image
from utils import centering_image, denorm_image


class Trainer:
    def __init__(self, GPU, data_dir, fixed_dir, fonts_num, batch_size, img_size):
        self.GPU = GPU
        self.data_dir = data_dir
        self.fixed_dir = fixed_dir
        self.fonts_num = fonts_num
        self.batch_size = batch_size
        self.img_size = img_size

        self.embeddings = torch.load(os.path.join(fixed_dir, "EMBEDDINGS.pkl"))
        self.embedding_num = self.embeddings.shape[0]
        self.embedding_dim = self.embeddings.shape[3]

        self.fixed_source = torch.load(os.path.join(fixed_dir, "fixed_source.pkl"))
        self.fixed_target = torch.load(os.path.join(fixed_dir, "fixed_target.pkl"))
        self.fixed_label = torch.load(os.path.join(fixed_dir, "fixed_label.pkl"))

        self.data_provider = TrainDataProvider(self.data_dir)
        self.total_batches = self.data_provider.compute_total_batch_num(self.batch_size)
        print("total batches:", self.total_batches)

    def train(
        self,
        max_epoch,
        schedule,
        save_path,
        to_model_path,
        lr=0.001,
        log_step=100,
        sample_step=350,
        fine_tune=False,
        flip_labels=False,
        restore=None,
        from_model_path=False,
        with_charid=False,
        freeze_encoder=False,
        save_nrow=8,
        model_save_epoch=None,
        resize_fix=90,
    ):
        # Fine Tuning coefficient
        if not fine_tune:
            L1_penalty, Lconst_penalty = 100, 15
        else:
            L1_penalty, Lconst_penalty = 500, 1000

        # Get Models
        En_ST = Encoder()
        De_ST = Decoder()
        D_ST = Discriminator(category_num=self.fonts_num)

        En_TS = Encoder()
        De_TS = Decoder()
        D_TS = Discriminator(category_num=self.fonts_num)

        if self.GPU:
            En_ST.cuda()
            De_ST.cuda()
            D_ST.cuda()

            En_TS.cuda()
            De_TS.cuda()
            D_TS.cuda()

        # Use pre-trained Model
        # restore에 [encoder_ST_path, decoder_ST_path, discriminator_ST_path] 형태로 인자 넣기
        if restore:
            (
                encoder_ST_path,
                decoder_ST_path,
                discriminator_ST_path,
                encoder_TS_path,
                decoder_TS_path,
                discriminator_TS_path,
            ) = restore
            prev_epoch = int(encoder_ST_path.split("-")[0])
            En_ST.load_state_dict(
                torch.load(os.path.join(from_model_path, encoder_ST_path))
            )
            De_ST.load_state_dict(
                torch.load(os.path.join(from_model_path, decoder_ST_path))
            )
            D_ST.load_state_dict(
                torch.load(os.path.join(from_model_path, discriminator_ST_path))
            )
            En_TS.load_state_dict(
                torch.load(os.path.join(from_model_path, encoder_TS_path))
            )
            De_TS.load_state_dict(
                torch.load(os.path.join(from_model_path, decoder_TS_path))
            )
            D_TS.load_state_dict(
                torch.load(os.path.join(from_model_path, discriminator_TS_path))
            )
            print("%d epoch trained model has restored" % prev_epoch)
        else:
            prev_epoch = 0
            print("New model training start")

        # L1 loss, binary real/fake loss, category loss, constant loss
        if self.GPU:
            l1_criterion = nn.L1Loss(size_average=True).cuda()
            bce_criterion = nn.BCEWithLogitsLoss(size_average=True).cuda()
            mse_criterion = nn.MSELoss(size_average=True).cuda()
        else:
            l1_criterion = nn.L1Loss(size_average=True)
            bce_criterion = nn.BCEWithLogitsLoss(size_average=True)
            mse_criterion = nn.MSELoss(size_average=True)

        # optimizer
        if freeze_encoder:
            G_ST_parameters = list(De_ST.parameters())
            G_TS_parameters = list(De_TS.parameters())
        else:
            G_ST_parameters = list(En_ST.parameters()) + list(De_ST.parameters())
            G_TS_parameters = list(En_TS.parameters()) + list(De_TS.parameters())
        G_ST_optimizer = torch.optim.Adam(G_ST_parameters, betas=(0.5, 0.999))
        D_ST_optimizer = torch.optim.Adam(D_ST.parameters(), betas=(0.5, 0.999))

        G_TS_optimizer = torch.optim.Adam(G_TS_parameters, betas=(0.5, 0.999))
        D_TS_optimizer = torch.optim.Adam(D_TS.parameters(), betas=(0.5, 0.999))

        # losses lists
        (
            l1_losses,
            const_losses,
            category_losses,
            d_losses,
            g_losses,
            l1_losses_TS,
            const_losses_TS,
            category_losses_TS,
            d_losses_TS,
            g_losses_TS,
            cycle_consistency_losses,
        ) = ([], [], [], [], [], [], [], [], [], [], [])

        # training
        # wandb
        wandb.init(project="big-project font_generator")
        count = 0
        for epoch in range(max_epoch):
            if (epoch + 1) % schedule == 0:
                updated_lr = max(lr / 2, 0.0002)
                for param_group in D_ST_optimizer.param_groups:
                    param_group["lr"] = updated_lr
                for param_group in G_ST_optimizer.param_groups:
                    param_group["lr"] = updated_lr
                for param_group in D_TS_optimizer.param_groups:
                    param_group["lr"] = updated_lr
                for param_group in G_TS_optimizer.param_groups:
                    param_group["lr"] = updated_lr
                if lr != updated_lr:
                    print("decay learning rate from %.5f to %.5f" % (lr, updated_lr))
                lr = updated_lr

            train_batch_iter = self.data_provider.get_train_iter(
                self.batch_size, with_charid=with_charid
            )
            for i, batch in enumerate(train_batch_iter):
                if with_charid:
                    font_ids, char_ids, batch_images = batch
                else:
                    font_ids, batch_images = batch
                embedding_ids = font_ids
                if self.GPU:
                    batch_images = batch_images.cuda()
                if flip_labels:
                    np.random.shuffle(embedding_ids)

                # target / source images
                real_target = batch_images[:, 0, :, :]
                real_target = real_target.view(
                    [self.batch_size, 1, self.img_size, self.img_size]
                )
                real_source = batch_images[:, 1, :, :]
                real_source = real_source.view(
                    [self.batch_size, 1, self.img_size, self.img_size]
                )

                # centering
                for idx, (image_S, image_T) in enumerate(zip(real_source, real_target)):
                    image_S = (
                        image_S.cpu()
                        .detach()
                        .numpy()
                        .reshape(self.img_size, self.img_size)
                    )
                    image_S = centering_image(image_S, resize_fix=90)
                    real_source[idx] = torch.tensor(image_S).view(
                        [1, self.img_size, self.img_size]
                    )
                    image_T = (
                        image_T.cpu()
                        .detach()
                        .numpy()
                        .reshape(self.img_size, self.img_size)
                    )
                    image_T = centering_image(image_T, resize_fix=resize_fix)
                    real_target[idx] = torch.tensor(image_T).view(
                        [1, self.img_size, self.img_size]
                    )

                # generate fake image form source image and for cycleGAN generate fake image from target image
                fake_target, encoded_source, _ = Generator(
                    real_source,
                    En_ST,
                    De_ST,
                    self.embeddings,
                    embedding_ids,
                    GPU=self.GPU,
                    encode_layers=True,
                )

                fake_source, encoded_target, _ = Generator(
                    real_target,
                    En_TS,
                    De_TS,
                    self.embeddings,
                    embedding_ids,
                    GPU=self.GPU,
                    encode_layers=True,
                )

                real = torch.cat([real_source, real_target], dim=1)
                fake_ST = torch.cat([real_source, fake_target], dim=1)
                fake_TS = torch.cat([fake_source, real_target], dim=1)

                # Scoring with Discriminator_ST
                real_score, real_score_logit, real_cat_logit = D_ST(real)
                fake_score, fake_score_logit, fake_cat_logit = D_ST(fake_ST)

                # Score with Discriminator_TS
                real_score_TS, real_score_logit_TS, real_cat_logit_TS = D_TS(real)
                fake_score_TS, fake_score_logit_TS, fake_cat_logit_TS = D_TS(fake_TS)

                # Get encoded fake target image to calculate constant loss
                encoded_fake = En_ST(fake_target)[0]
                const_loss = Lconst_penalty * mse_criterion(
                    encoded_source, encoded_fake
                )

                # Get encoded fake source image to calculate constant loss
                encoded_real = En_TS(fake_source)[0]
                const_loss_TS = Lconst_penalty * mse_criterion(
                    encoded_target, encoded_real
                )

                # category loss
                real_category = torch.from_numpy(
                    np.eye(self.fonts_num)[embedding_ids]
                ).float()

                # category loss2
                real_category_TS = torch.from_numpy(
                    np.eye(self.fonts_num)[embedding_ids]
                ).float()

                if self.GPU:
                    real_category = real_category.cuda()
                    real_category_TS = real_category_TS.cuda()

                real_category_loss = bce_criterion(real_cat_logit, real_category)
                fake_category_loss = bce_criterion(fake_cat_logit, real_category)

                # category loss2 for TS real is 100
                real_category_loss_TS = bce_criterion(
                    real_cat_logit_TS, real_category_TS
                )
                fake_category_loss_TS = bce_criterion(
                    fake_cat_logit_TS, real_category_TS
                )

                category_loss = 0.5 * (real_category_loss + fake_category_loss)
                category_loss_TS = 0.5 * (real_category_loss_TS + fake_category_loss_TS)

                # labels
                if self.GPU:
                    one_labels = torch.ones([self.batch_size, 1]).cuda()
                    zero_labels = torch.zeros([self.batch_size, 1]).cuda()

                    # category loss2 for TS real is 100
                    one_labels_TS = torch.ones([self.batch_size, 1]).cuda()
                    zero_labels_TS = torch.zeros([self.batch_size, 1]).cuda()

                else:
                    one_labels = torch.ones([self.batch_size, 1])
                    zero_labels = torch.zeros([self.batch_size, 1])

                    # category loss2 for TS real is 100
                    one_labels_TS = torch.ones([self.batch_size, 1])
                    zero_labels_TS = torch.zeros([self.batch_size, 1])

                # binary loss - T/F
                real_binary_loss = bce_criterion(real_score_logit, one_labels)
                fake_binary_loss = bce_criterion(fake_score_logit, zero_labels)
                binary_loss = real_binary_loss + fake_binary_loss

                # binary loss - T/F
                real_binary_loss_TS = bce_criterion(real_score_logit_TS, one_labels_TS)
                fake_binary_loss_TS = bce_criterion(fake_score_logit_TS, zero_labels_TS)
                binary_loss_TS = real_binary_loss_TS + fake_binary_loss_TS

                # L1 loss between real and fake images
                l1_loss = L1_penalty * l1_criterion(real_target, fake_target)
                l1_loss_TS = L1_penalty * l1_criterion(real_source, fake_source)

                # cheat loss for generator to fool discriminator
                cheat_loss = bce_criterion(fake_score_logit, one_labels)
                cheat_loss_TS = bce_criterion(fake_score_logit_TS, one_labels_TS)

                # cycle consistency loss
                cycle_consistency_loss = L1_penalty * l1_loss_TS + L1_penalty * l1_loss

                # g_loss, d_loss
                g_loss = (
                    cheat_loss
                    + l1_loss
                    + fake_category_loss
                    + const_loss
                    + cycle_consistency_loss
                )
                d_loss = binary_loss + category_loss

                # g_loss, d_loss
                g_loss_TS = (
                    cheat_loss_TS
                    + l1_loss_TS
                    + fake_category_loss_TS
                    + const_loss_TS
                    + cycle_consistency_loss
                )
                d_loss_TS = binary_loss_TS + category_loss_TS

                # train Discriminator
                D_ST.zero_grad()
                D_TS.zero_grad()
                d_loss.backward(retain_graph=True)
                D_ST_optimizer.step()
                D_TS_optimizer.step()

                # train Generator
                En_ST.zero_grad()
                De_ST.zero_grad()
                En_TS.zero_grad()
                De_TS.zero_grad()
                g_loss.backward(retain_graph=True)
                G_ST_optimizer.step()
                G_TS_optimizer.step()

                # loss data
                l1_losses.append(int(l1_loss.data))
                const_losses.append(int(const_loss.data))
                category_losses.append(int(category_loss.data))
                d_losses.append(int(d_loss.data))
                g_losses.append(int(g_loss.data))

                cycle_consistency_losses.append(int(cycle_consistency_loss.data))

                # loss data
                l1_losses_TS.append(int(l1_loss_TS.data))
                const_losses_TS.append(int(const_loss_TS.data))
                category_losses_TS.append(int(category_loss_TS.data))
                d_losses_TS.append(int(d_loss_TS.data))
                g_losses_TS.append(int(g_loss_TS.data))

                # logging
                if (i + 1) % log_step == 0:
                    time_ = time.time()
                    time_stamp = datetime.datetime.fromtimestamp(time_).strftime(
                        "%H:%M:%S"
                    )
                    log_format = (
                        "Epoch [%d/%d], step [%d/%d], l1_loss: %.4f, d_loss: %.4f, g_loss: %.4f, l1_loss_TS: %.4f, d_loss_TS: %.4f, g_loss_TS: %.4f, cycle_consistency_loss: %.4f"
                        % (
                            int(prev_epoch) + epoch + 1,
                            int(prev_epoch) + max_epoch,
                            i + 1,
                            self.total_batches,
                            l1_loss.item(),
                            d_loss.item(),
                            g_loss.item(),
                            l1_loss_TS.item(),
                            d_loss_TS.item(),
                            g_loss_TS.item(),
                            cycle_consistency_loss.item(),
                        )
                    )
                    print(time_stamp, log_format)
                    wandb.log(
                        {
                            "l1_loss": l1_loss.item(),
                            "d_loss": d_loss.item(),
                            "g_loss": g_loss.item(),
                            "l1_loss_TS": l1_loss_TS.item(),
                            "d_loss_TS": d_loss_TS.item(),
                            "g_loss_TS": g_loss_TS.item(),
                            "cycle_consistency_loss": cycle_consistency_loss.item(),
                        }
                    )

                # save image
                if (i + 1) % sample_step == 0:
                    fixed_fake_images = Generator(
                        self.fixed_source,
                        En_ST,
                        De_ST,
                        self.embeddings,
                        self.fixed_label,
                        GPU=self.GPU,
                    )[0]
                    save_image(
                        denorm_image(fixed_fake_images.data),
                        os.path.join(
                            save_path,
                            "fake_samples-%d-%d.png"
                            % (int(prev_epoch) + epoch + 1, i + 1),
                        ),
                        nrow=save_nrow,
                        pad_value=255,
                    )

            if not model_save_epoch:
                model_save_epoch = 5
            if (epoch + 1) % model_save_epoch == 0:
                now = datetime.datetime.now()
                now_date = now.strftime("%m%d")
                now_time = now.strftime("%H:%M")
                torch.save(
                    En_ST.state_dict(),
                    os.path.join(
                        to_model_path,
                        "%d-%s-%s-Encoder.pkl"
                        % (int(prev_epoch) + epoch + 1, now_date, now_time),
                    ),
                )
                torch.save(
                    De_ST.state_dict(),
                    os.path.join(
                        to_model_path,
                        "%d-%s-%s-Decoder.pkl"
                        % (int(prev_epoch) + epoch + 1, now_date, now_time),
                    ),
                )
                torch.save(
                    D_ST.state_dict(),
                    os.path.join(
                        to_model_path,
                        "%d-%s-%s-Discriminator.pkl"
                        % (int(prev_epoch) + epoch + 1, now_date, now_time),
                    ),
                )
                torch.save(
                    En_TS.state_dict(),
                    os.path.join(
                        to_model_path,
                        "%d-%s-%s-Encoder_TS.pkl"
                        % (int(prev_epoch) + epoch + 1, now_date, now_time),
                    ),
                ),
                torch.save(
                    De_TS.state_dict(),
                    os.path.join(
                        to_model_path,
                        "%d-%s-%s-Decoder_TS.pkl"
                        % (int(prev_epoch) + epoch + 1, now_date, now_time),
                    ),
                ),
                torch.save(
                    D_TS.state_dict(),
                    os.path.join(
                        to_model_path,
                        "%d-%s-%s-Discriminator_TS.pkl"
                        % (int(prev_epoch) + epoch + 1, now_date, now_time),
                    ),
                )

        # save model
        total_epoch = int(prev_epoch) + int(max_epoch)
        end = datetime.datetime.now()
        end_date = end.strftime("%m%d")
        end_time = end.strftime("%H:%M")
        torch.save(
            En_ST.state_dict(),
            os.path.join(
                to_model_path,
                "%d-%s-%s-Encoder.pkl" % (total_epoch, end_date, end_time),
            ),
        )
        torch.save(
            De_ST.state_dict(),
            os.path.join(
                to_model_path,
                "%d-%s-%s-Decoder.pkl" % (total_epoch, end_date, end_time),
            ),
        )
        torch.save(
            D_ST.state_dict(),
            os.path.join(
                to_model_path,
                "%d-%s-%s-Discriminator.pkl" % (total_epoch, end_date, end_time),
            ),
        )
        torch.save(
            En_TS.state_dict(),
            os.path.join(
                to_model_path,
                "%d-%s-%s-Encoder_TS.pkl" % (total_epoch, end_date, end_time),
            ),
        ),
        torch.save(
            De_TS.state_dict(),
            os.path.join(
                to_model_path,
                "%d-%s-%s-Decoder_TS.pkl" % (total_epoch, end_date, end_time),
            ),
        ),
        torch.save(
            D_TS.state_dict(),
            os.path.join(
                to_model_path,
                "%d-%s-%s-Discriminator_TS.pkl" % (total_epoch, end_date, end_time),
            ),
        )

        losses = [
            l1_losses,
            const_losses,
            category_losses,
            d_losses,
            g_losses,
            l1_losses_TS,
            const_losses_TS,
            category_losses_TS,
            d_losses_TS,
            g_losses_TS,
            cycle_consistency_losses,
        ]
        torch.save(losses, os.path.join(to_model_path, "%d-losses.pkl" % total_epoch))

        return (
            l1_losses,
            const_losses,
            category_losses,
            d_losses,
            g_losses,
            l1_losses_TS,
            const_losses_TS,
            category_losses_TS,
            d_losses_TS,
            g_losses_TS,
            cycle_consistency_losses,
        )


if __name__ == "__main__":
    Trainer = Trainer(
        GPU=torch.cuda.is_available(),
        data_dir="./data",
        fixed_dir="./data",
        fonts_num=25,
        batch_size=32,
        img_size=128,
    )

    # train
    Trainer.train(
        max_epoch=30,
        schedule=20,
        save_path="./fixed_fake",
        to_model_path="./checkpoint",
        lr=0.001,
        log_step=700,
        sample_step=700,
        fine_tune=False,
        flip_labels=False,
        restore=None,
        from_model_path=False,
        with_charid=True,
        freeze_encoder=False,
        save_nrow=8,
        model_save_epoch=5,
        resize_fix=90,
    )

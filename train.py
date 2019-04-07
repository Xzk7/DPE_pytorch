from model import Generator, Discriminator
from parameter import NetInfo

import os
import time
import torch
import datetime
import numpy as np
from tqdm import tqdm

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F

from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'

class Trainer_DPE(object):
    def __init__(self, data_loader, config):
        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model
        self.model_use_debug = config.model_use_debug
        self.num_exp = config.num_exp
        self.num_gpu = config.num_gpu

        # Model hyper-parameters
        self.imsize = config.imsize
        self.batch_size = config.batch_size
        self.parallel = config.parallel
        self.gpus = config.gpus

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.netD_init_method = config.netD_init_method
        self.netD_init_weight = config.netD_init_weight
        self.netD_lr = config.netD_lr
        self.netD_lr_decay = config.netD_lr_decay
        self.netD_lr_decay_epoch = config.netD_lr_decay_epoch
        self.netD_regularization_weight = config.netD_regularization_weight
        self.netG_init_weight = config.netG_init_weight
        self.netD_times = -config.netD_init_times
        self.netD_change_times_1 = config.netD_times
        self.netD_change_times_2 = config.netD_times
        self.netD_times_grow = config.netD_times_grow
        self.netD_buffer_times = config.netD_buffer_times
        self.netG_lr = config.netG_lr
        self.netG_lr_decay = config.netG_lr_decay
        self.netG_lr_decay_epoch = config.netG_lr_decay_epoch
        self.netG_regularization_weight = config.netG_regularization_weight
        self.loss_source_data_term = config.loss_source_data_term
        self.loss_source_data_term_weight = config.loss_source_data_term_weight
        self.loss_constant_term = config.loss_constant_term
        self.loss_constant_term_weight = config.loss_constant_term_weight
        self.loss_photorealism_is_our = config.loss_photorealism_is_our
        self.loss_wgan_lambda = config.loss_wgan_lambda
        self.loss_wgan_lambda_grow = config.loss_wgan_lambda_grow
        self.loss_wgan_lambda_ignore = config.loss_wgan_lambda_ignore
        self.loss_wgan_use_g_to_one = config.loss_wgan_use_g_to_one
        self.loss_wgan_gp_times = config.loss_wgan_gp_times
        self.loss_wgan_gp_use_all = config.loss_wgan_gp_use_all
        self.loss_wgan_gp_bound = config.loss_wgan_gp_bound
        self.loss_wgan_gp_mv_decay = config.loss_wgan_gp_mv_decay
        self.loss_data_term_use_local_weight = config.loss_data_term_use_local_weight
        self.loss_constant_term_use_local_weight = config.loss_constant_term_use_local_weight
        self.data_csr_buffer_size = config.data_csr_buffer_size
        self.sys_use_all_gpu_memory = config.sys_use_all_gpu_memory
        self.loss_pr = config.loss_pr
        self.loss_heavy = config.loss_heavy
        self.data_augmentation_size = config.data_augmentation_size
        self.data_use_random_pad = config.data_use_random_pad
        self.data_train_batch_size = config.data_train_batch_size
        self.load_previous_exp = config.load_previous_exp
        self.load_previous_epoch = config.load_previous_epoch
        self.process_run_first_testing_epoch = config.process_run_first_testing_epoch
        self.process_write_test_img_count = config.process_write_test_img_count
        self.process_train_log_interval_epoch = config.process_train_log_interval_epoch
        self.process_test_log_interval_epoch = config.process_test_log_interval_epoch
        self.process_max_epoch = config.process_max_epoch
        self.gp_weight1 = config.loss_wgan_lambda
        self.gp_weight2 = config.loss_wgan_lambda

        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version
        self.step_per_epoch = len(self.data_loader)

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.netD_wgan_gp_mvavg_1 = 0
        self.netD_wgan_gp_mvavg_2 = 0
        self.netD_update_buffer_1 = 0
        self.netD_update_buffer_2 = 0

        self.G_training = False

        self.config = config

        print('build_model...')
        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            print('load_pretrained_model...')
            self.load_pretrained_model()


    def train(self):
        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 1

        print('Start   ======  training...')
        start_time = time.time()
        for epoch in range(start, self.process_max_epoch+1):
            # for netD
            total_wd_A_loss = 0.0
            total_wd_B_loss = 0.0
            total_gp1 = 0.0
            total_gp2 = 0.0
            total_d_loss = 0.0
            # for netG
            total_AG_loss = 0.0
            total_I_loss = 0.0
            total_C_loss = 0.0
            total_g_loss = 0.0
            G_epoch = 0
            i = 0
            for (real_A, real_B) in tqdm(self.data_loader):
                i = i+1
                step = (epoch-1)*self.step_per_epoch + i

                real_A = self.preprocess(real_A)
                real_B = self.preprocess(real_B)

                self.D1.train()
                self.G1.train()
                self.D2.train()
                self.G2.train()

                # ================== Train D net ================== #
                fake_A, fake_A_list = self.G2(dict(img=real_B, retouched=False))
                fake_B, fake_B_list = self.G1(dict(img=real_A, retouched=False))
                rec_B, rec_B_list = self.G1(dict(img=torch.clamp(fake_A, -1, 1), retouched=True))
                rec_A, rec_A_list = self.G2(dict(img=torch.clamp(fake_B, -1, 1), retouched=True))
                d_fake_A, d_fake_A_list = self.D1(fake_A)
                d_real_A, d_real_A_list = self.D1(real_A)
                d_fake_B, d_fake_B_list = self.D2(fake_B)
                d_real_B, d_real_B_list = self.D2(real_B)

                gradient_penalty1 = self.wgan_gp(fake_A, real_A, True) * self.gp_weight1
                gradient_penalty2 = self.wgan_gp(fake_B, real_B, False) * self.gp_weight2

                wd_A = -torch.mean(d_real_A) + torch.mean(d_fake_A)
                wd_B = -torch.mean(d_fake_B) + torch.mean(d_real_B)
                netD_train_loss = wd_A + wd_B
                netD_loss = netD_train_loss + gradient_penalty1 + gradient_penalty2
                d_loss = -netD_loss

                # for data analysis
                total_wd_A_loss += -wd_A.item()
                total_wd_B_loss += -wd_B.item()
                total_gp1 += -gradient_penalty1.item()
                total_gp2 += -gradient_penalty2.item()
                total_d_loss += d_loss.item()

                self.reset_grad()
                d_loss.backward()
                self.d1_optimizer.step()
                self.d2_optimizer.step()

                if not (step < self.loss_wgan_lambda_ignore):
                    self.netD_wgan_gp_mvavg_1 = self.netD_wgan_gp_mvavg_1 * self.loss_wgan_gp_mv_decay + (
                                -gradient_penalty1.item() / self.gp_weight1) * (1 - self.loss_wgan_gp_mv_decay)
                    self.netD_wgan_gp_mvavg_2 = self.netD_wgan_gp_mvavg_2 * self.loss_wgan_gp_mv_decay + (
                                -gradient_penalty2.item() / self.gp_weight2) * (1 - self.loss_wgan_gp_mv_decay)

                if self.netD_update_buffer_1 == 0 and self.netD_wgan_gp_mvavg_1 > self.loss_wgan_gp_bound:
                    self.gp_weight1 = self.gp_weight1 * self.loss_wgan_lambda_grow
                    self.netD_change_times_1 = self.netD_change_times_1 * self.netD_times_grow
                    self.netD_update_buffer_1 = self.netD_buffer_times
                    self.netD_wgan_gp_mvavg_1 = 0
                self.netD_update_buffer_1 = 0 if self.netD_update_buffer_1 == 0 else self.netD_update_buffer_1 - 1

                if self.netD_update_buffer_2 == 0 and self.netD_wgan_gp_mvavg_2 > self.loss_wgan_gp_bound:
                    self.gp_weight2 = self.gp_weight2 * self.loss_wgan_lambda_grow
                    self.netD_change_times_2 = self.netD_change_times_2 * self.netD_times_grow
                    self.netD_update_buffer_2 = self.netD_buffer_times
                    self.netD_wgan_gp_mvavg_2 = 0
                self.netD_update_buffer_2 = 0 if self.netD_update_buffer_2 == 0 else self.netD_update_buffer_2 - 1

                # ================== Train G net  ================== #
                if self.netD_change_times_1 > 0 and self.netD_times >= 0 and self.netD_times % self.netD_change_times_1 == 0:
                    self.netD_times = 0
                    fake_A, fake_A_list = self.G2(dict(img=real_B, retouched=False))
                    fake_B, fake_B_list = self.G1(dict(img=real_A, retouched=False))
                    rec_B, rec_B_list = self.G1(dict(img=torch.clamp(fake_A, -1, 1), retouched=True))
                    rec_A, rec_A_list = self.G2(dict(img=torch.clamp(fake_B, -1, 1), retouched=True))
                    d_fake_A, d_fake_A_list = self.D1(fake_A)
                    d_real_A, d_real_A_list = self.D1(real_A)
                    d_fake_B, d_fake_B_list = self.D2(fake_B)
                    d_real_B, d_real_B_list = self.D2(real_B)

                    if self.loss_source_data_term_weight > 0:
                        if self.loss_source_data_term == 'l2':
                            train_data_term_1 = -self.img_L2_loss(fake_B, real_A, self.loss_data_term_use_local_weight) * self.loss_source_data_term_weight
                            train_data_term_2 = -self.img_L2_loss(fake_A, real_B, self.loss_data_term_use_local_weight) * self.loss_source_data_term_weight
                        elif self.loss_source_data_term == 'l1':
                            train_data_term_1 = -self.img_L1_loss(fake_B, real_A) * self.loss_source_data_term_weight
                            train_data_term_2 = -self.img_L1_loss(fake_A, real_B) * self.loss_source_data_term_weight
                        else:
                            assert False, 'not yet'

                    else:
                        train_data_term_1 = Variable(torch.FloatTensor([0.]).squeeze(-1))
                        train_data_term_2 = Variable(torch.FloatTensor([0.]).squeeze(-1))

                    if self.loss_constant_term_weight > 0:
                        if self.loss_constant_term == 'l2':
                            train_constant_term_1 = -self.img_L2_loss(rec_A, real_A, self.loss_constant_term_use_local_weight) * self.loss_constant_term_weight
                            train_constant_term_2 = -self.img_L2_loss(rec_B, real_B, self.loss_constant_term_use_local_weight) * self.loss_constant_term_weight
                        elif self.loss_source_data_term == 'l1':
                            train_constant_term_1 = -self.img_L1_loss(rec_A, real_A) * self.loss_constant_term_weight
                            train_constant_term_2 = -self.img_L1_loss(rec_B, real_B) * self.loss_constant_term_weight
                        else:
                            assert False, 'not yet'

                    else:
                        train_constant_term_1 = Variable(torch.FloatTensor([0.]).squeeze(-1))
                        train_constant_term_2 = Variable(torch.FloatTensor([0.]).squeeze(-1))

                    netG_train_loss = torch.mean(d_fake_B) - torch.mean(d_fake_A)
                    netG_loss = netG_train_loss + train_data_term_1 + train_data_term_2 + train_constant_term_1 + train_constant_term_2
                    g_loss = -netG_loss

                    # for data analysis
                    total_AG_loss += -netG_train_loss.item()
                    total_I_loss += - (train_data_term_1.item() + train_data_term_2.item())
                    total_C_loss += - (train_constant_term_1.item() + train_constant_term_2.item())
                    total_g_loss += g_loss.item()
                    G_epoch += 1
                    self.G_training = True

                    self.reset_grad()
                    g_loss.backward()
                    self.g1_optimizer.step()
                    self.g2_optimizer.step()

                else:
                    self.G_training = False

                self.netD_times += 1

                if (step + 1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed [{}], epoch [{}/{}], i [{}/{}]".format(elapsed, epoch, self.process_max_epoch, (i + 1), self.step_per_epoch))
                    print("Discriminator training")
                    print("avg_wd_A_loss: {:.4f}, avg_wd_B_loss: {:.4f}, avg_gp1: {:.4f}, avg_gp2: {:.4f}, avg_d_loss: {:.4f}".format(
                        total_wd_A_loss/i, total_wd_B_loss/i, total_gp1/i, total_gp2/i, total_d_loss/i
                    ))
                    if self.use_tensorboard:
                        self.writer.add_scalar('data/avg_wd_A_loss', total_wd_A_loss/i, (step + 1))
                        self.writer.add_scalar('data/avg_wd_B_loss', total_wd_B_loss/i, (step + 1))
                        self.writer.add_scalar('data/avg_gp1', total_gp1/i, (step + 1))
                        self.writer.add_scalar('data/avg_gp2', total_gp2/i, (step + 1))
                        self.writer.add_scalar('data/avg_d_loss', total_d_loss/i, (step + 1))
                if self.G_training:
                    print("Generator training")
                    print(
                        "avg_AG_loss: {:.4f}, avg_I_loss: {:.4f}, avg_C_loss: {:.4f}, avg_g_loss: {:.4f}".format(
                            total_AG_loss / G_epoch, total_I_loss / G_epoch, total_C_loss / G_epoch,
                            total_g_loss / G_epoch
                        ))
                    if self.use_tensorboard:
                        self.writer.add_scalar('data/avg_AG_loss', total_AG_loss / G_epoch, (step+1))
                        self.writer.add_scalar('data/avg_I_loss', total_I_loss / G_epoch, (step+1))
                        self.writer.add_scalar('data/avg_C_loss', total_C_loss / G_epoch, (step+1))
                        self.writer.add_scalar('data/avg_g_loss', total_g_loss / G_epoch, (step+1))

                # Sample images
                if (step+1) % self.sample_step == 0:
                    print('Sample images {}.png'.format(step + 1))
                    fake_A, fake_A_list = self.G2(dict(img=real_B, retouched=False))
                    fake_B, fake_B_list = self.G1(dict(img=real_A, retouched=False))
                    rec_B, rec_B_list = self.G1(dict(img=torch.clamp(fake_A, -1, 1), retouched=True))
                    rec_A, rec_A_list = self.G2(dict(img=torch.clamp(fake_B, -1, 1), retouched=True))
                    out_A = torch.cat([real_A, fake_B, rec_A], 0)
                    save_image(denorm(out_A.data), os.path.join(self.sample_path, 'imgA_{}.png'.format(step + 1)))
                    out_B = torch.cat([real_B, fake_A, rec_B], 0)
                    save_image(denorm(out_B.data), os.path.join(self.sample_path, 'imgB_{}.png'.format(step + 1)))

                if epoch % self.model_save_step == 0:
                    torch.save(self.G1.state_dict(),
                               os.path.join(self.model_save_path, 'G1_{}.pth'.format(epoch + 1)))
                    torch.save(self.D1.state_dict(),
                               os.path.join(self.model_save_path, 'D1_{}.pth'.format(epoch + 1)))
                    torch.save(self.G2.state_dict(),
                               os.path.join(self.model_save_path, 'G2_{}.pth'.format(epoch + 1)))
                    torch.save(self.D2.state_dict(),
                               os.path.join(self.model_save_path, 'D2_{}.pth'.format(epoch + 1)))

                if epoch >= self.netD_lr_decay_epoch:
                    self.d1_optimizer.param_groups[0]['lr'] = self.d1_optimizer.param_groups[0]['lr'] - (self.config.netD_lr / self.config.netD_lr_decay)
                    self.d2_optimizer.param_groups[0]['lr'] = self.d2_optimizer.param_groups[0]['lr'] - (self.config.netD_lr / self.config.netD_lr_decay)
                if epoch >= self.netG_lr_decay_epoch:
                    self.g1_optimizer.param_groups[0]['lr'] = self.g1_optimizer.param_groups[0]['lr'] - (self.config.netG_lr / self.config.netG_lr_decay)
                    self.g2_optimizer.param_groups[0]['lr'] = self.g2_optimizer.param_groups[0]['lr'] - (self.config.netG_lr / self.config.netG_lr_decay)

    def build_model(self):
        netG = NetInfo("netG")
        self.G1 = Generator(netG)
        self.G2 = Generator(netG)
        self.G2.load_state_dict(self.G1.state_dict())
        self.G1 = self.G1.to(self.device)
        self.G2 = self.G2.to(self.device)
        netD = NetInfo("netD")
        self.D1 = Discriminator(netD)
        self.D2 = Discriminator(netD)
        self.D2.load_state_dict(self.D1.state_dict())
        self.D1 = self.D1.to(self.device)
        self.D2 = self.D2.to(self.device)

        if self.parallel:
            print('use parallel...')
            print('gpuids ', self.gpus)
            gpus = [int(i) for i in self.gpus.split(',')]

            self.G1 = nn.DataParallel(self.G1, device_ids=gpus).cuda()
            self.G2 = nn.DataParallel(self.G2, device_ids=gpus).cuda()
            self.D1 = nn.DataParallel(self.D1, device_ids=gpus).cuda()
            self.D2 = nn.DataParallel(self.D2, device_ids=gpus).cuda()

        model_parameters = filter(lambda p: p.requires_grad, self.G1.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('|  Number of G1 Parameters: ' + str(params))
        model_parameters = filter(lambda p: p.requires_grad, self.G2.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('|  Number of G2 Parameters: ' + str(params))
        model_parameters = filter(lambda p: p.requires_grad, self.D1.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('|  Number of D1 Parameters: ' + str(params))
        model_parameters = filter(lambda p: p.requires_grad, self.D2.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('|  Number of D2 Parameters: ' + str(params))

        self.g1_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G1.parameters()), self.netG_lr,
                                            [self.beta1, self.beta2], weight_decay=self.netG_regularization_weight)
        self.d1_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D1.parameters()), self.netD_lr,
                                            [self.beta1, self.beta2], weight_decay=self.netD_regularization_weight)
        self.g2_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G2.parameters()), self.netG_lr,
                                             [self.beta1, self.beta2], weight_decay=self.netG_regularization_weight)
        self.d2_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D2.parameters()), self.netD_lr,
                                             [self.beta1, self.beta2], weight_decay=self.netD_regularization_weight)

    def build_tensorboard(self):
        from tensorboardX import SummaryWriter
        # from logger import Logger
        # self.logger = Logger(self.log_path)

        tf_logs_path = os.path.join(self.log_path, 'tf_logs')
        self.writer = SummaryWriter(log_dir=tf_logs_path)

    def load_pretrained_model(self):
        self.G1.load_state_dict(torch.load(os.path.join(
            self.model_save_path, 'G1_{}.pth'.format(self.pretrained_model))))
        self.D1.load_state_dict(torch.load(os.path.join(
            self.model_save_path, 'D1_{}.pth'.format(self.pretrained_model))))
        self.G2.load_state_dict(torch.load(os.path.join(
            self.model_save_path, 'G2_{}.pth'.format(self.pretrained_model))))
        self.D2.load_state_dict(torch.load(os.path.join(
            self.model_save_path, 'D2_{}.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d1_optimizer.zero_grad()
        self.g1_optimizer.zero_grad()
        self.d2_optimizer.zero_grad()
        self.g2_optimizer.zero_grad()

    def preprocess(self, img_512):
        img_512 = img_512.type(torch.FloatTensor)
        img_512 = Variable(img_512.cuda() if self.parallel else img_512)
        return img_512

    def wgan_gp(self, fake_data, real_data, net):
        #fake_data = fake_data.reshape(self.batch_size, -1)
        #real_data = real_data.reshape(self.batch_size, -1)
        alpha = np.random.uniform(low=0., high=1., size=(self.batch_size,1,1,1))
        alpha = np.float32(alpha)
        alpha = torch.from_numpy(alpha).to(self.device).expand_as(real_data)
        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)
        # interpolates_D = interpolates.reshape(self.batch_size, 3, self.imsize, self.imsize)
        if (net==True):
            output, output_list = self.D1(interpolates)
        else:
            output, output_list = self.D2(interpolates)
        gradients = torch.autograd.grad(output, interpolates, grad_outputs=torch.ones_like(output).to(self.device), retain_graph=True, create_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        slopes = torch.sqrt(torch.sum(gradients**2, dim=1))
        if self.loss_wgan_use_g_to_one:
            gradient_penalty = -torch.mean((slopes-1.)**2)
        else:
            zeros = torch.zeros_like(slopes)
            # print(slopes)
            gradient_penalty = -torch.mean(torch.max(zeros, slopes-1.))
            # print(gradient_penalty)
        return gradient_penalty

    def img_L1_loss(self, img1, img2):
        return torch.mean(torch.abs(torch.sub(img1, img2)))

    def img_L2_loss(self, img1, img2, use_local_weight):
        if use_local_weight==True:
            w = -torch.log(torch.FloatTensor(img2)+torch.exp(torch.FloatTensor([-99.]))) + 1
            w = w**2
            return torch.mean(w * (torch.sub(img1, img2)**2))
        else:
            return torch.mean(torch.sub(img1, img2)**2)

import os
import sys
import time
from datetime import datetime
from skimage import measure
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import logger
from cnn_owm import RED_CNN_OWM

matplotlib.use('TkAgg')
dtype = torch.FloatTensor

Logger = logger.Logger('./solver_iowm_results.txt')


class Solver(object):
    def __init__(self, args, out_ch):
        self.save_path = args.save_path      # the path of model
        self.num_epochs = args.epochs        # the total epochs
        self.print_iters = args.print_iters  # the frequence of print
        self.decay_iters = args.decay_iters  # the learning rate decrease
        self.save_iters = args.save_iters    # how many to save model
        self.patch_size = args.patch_size    # the size of patchs
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig
        self.multi_gpu = args.multi_gpu
        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.REDCNN = RED_CNN_OWM(out_ch)
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print(torch.cuda.device_count())
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.REDCNN = nn.DataParallel(self.REDCNN, device_ids=[1, 2, 3, 0])
        self.REDCNN.to(self.device)

        self.lr = args.lr                    # learning rate
        self.criterion = nn.MSELoss()        # loss function
        self.optimizer = optim.Adam(self.REDCNN.parameters(), self.lr)

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max
        # define projection matrix
        with torch.no_grad():
            self.P_conv1 = torch.autograd.Variable(torch.eye(1 * 5 * 5).type(dtype)).to(self.device)
            self.P_conv2 = torch.autograd.Variable(torch.eye(out_ch * 5 * 5).type(dtype)).to(self.device)
            self.P_conv3 = torch.autograd.Variable(torch.eye(out_ch * 5 * 5).type(dtype)).to(self.device)
            self.P_conv4 = torch.autograd.Variable(torch.eye(out_ch * 5 * 5).type(dtype)).to(self.device)
            self.P_conv5 = torch.autograd.Variable(torch.eye(out_ch * 5 * 5).type(dtype)).to(self.device)
            self.P_dconv1 = torch.autograd.Variable(torch.eye(out_ch * 5 * 5).type(dtype)).to(self.device)
            self.P_dconv2 = torch.autograd.Variable(torch.eye(out_ch * 5 * 5).type(dtype)).to(self.device)
            self.P_dconv3 = torch.autograd.Variable(torch.eye(out_ch * 5 * 5).type(dtype)).to(self.device)
            self.P_dconv4 = torch.autograd.Variable(torch.eye(out_ch * 5 * 5).type(dtype)).to(self.device)
            self.P_dconv5 = torch.autograd.Variable(torch.eye(out_ch * 5 * 5).type(dtype)).to(self.device)

    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        torch.save(self.REDCNN.state_dict(), f)

    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            weights = torch.load(f)
            for k in torch.load(f):
                n = k
                state_d[n] = weights[k]
            self.REDCNN.load_state_dict(state_d)
        else:
            self.REDCNN.load_state_dict(torch.load(f))

    def lr_decay(self, epochs=100):
        lr = self.lr * 0.1**(epochs//100)
        # lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('low-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()

    def save_raw(self, x, y, pred, fig_name):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        # y_x = y - x
        # label_pred = y - pred
        # strs = os.path.join(self.save_path, 'fig', 'label_data_{}.raw'.format(fig_name))
        # y_x.tofile(strs)
        # strs = os.path.join(self.save_path, 'fig', 'label_pred_{}.raw'.format(fig_name))
        # label_pred.tofile(strs)
        strs = os.path.join(self.save_path, 'fig', 'old_{}.raw'.format(fig_name))
        x.tofile(strs)
        strs = os.path.join(self.save_path, 'fig', 'label_{}.raw'.format(fig_name))
        y.tofile(strs)
        strs = os.path.join(self.save_path, 'fig', 'pred_{}.raw'.format(fig_name))
        pred.tofile(strs)

    def _get_optimizer(self):
        lr = self.lr
        lr_owm = self.lr

        optimizer = torch.optim.Adam([{'params': self.REDCNN.conv_first.parameters(), 'lr': lr_owm},
                                      {'params': self.REDCNN.conv2.parameters(), 'lr': lr_owm},
                                      {'params': self.REDCNN.conv3.parameters(), 'lr': lr_owm},
                                      {'params': self.REDCNN.conv4.parameters(), 'lr': lr_owm},
                                      {'params': self.REDCNN.conv5.parameters(), 'lr': lr_owm},
                                      {'params': self.REDCNN.conv_1.parameters(), 'lr': lr_owm},
                                      {'params': self.REDCNN.conv_2.parameters(), 'lr': lr_owm},
                                      {'params': self.REDCNN.conv_3.parameters(), 'lr': lr_owm},
                                      {'params': self.REDCNN.conv_4.parameters(), 'lr': lr_owm},
                                      {'params': self.REDCNN.conv_t_last.parameters(), 'lr': lr_owm}
                                      ], lr=lr)

        return optimizer

    def pro_weight(self, p, x, w, alpha=1.0, immune=True, dconv=False, stride=1, kernel_size=5):
        if dconv:
            x = torch.mean(x, 0, True)
            _, F, HH, WW = w.shape
            N, C, H, W = x.shape
            height = H + 2 * (HH - 1) + (H - 1) * (stride - 1)
            width = W + 2 * (WW - 1) + (W - 1) * (stride - 1)
            fill_x = torch.zeros(N, C, height, width).to(self.device)
            fill_x[:, :, HH - 1:height - HH + 1:stride, WW - 1:width - WW + 1:stride] = x
            x = fill_x.detach()
            N, C, H, W = x.shape
            S = kernel_size
            Ho = int(1 + (H - HH) / S)
            Wo = int(1 + (W - WW) / S)
        else:
            x = torch.mean(x, 0, True)
            N, C, H, W = x.shape
            F, _, HH, WW = w.shape
            S = kernel_size
            Ho = int(1 + (H - HH) / S)
            Wo = int(1 + (W - WW) / S)
        if immune:
            for i in range(Ho):
                for j in range(Wo):
                    r = x[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1)
                    k = torch.mm(p, torch.t(r))
                    deltap = torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k))
                    tmp_P = p - deltap
                    p = tmp_P.detach()
        if dconv:
            # w_tmp = w.grad.data.clone()
            w.grad.data = torch.mm(w.grad.data.permute(1, 0, 2, 3).contiguous().view(F, -1), torch.t(p.data)).view_as(w.grad.data.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
            # if not torch.allclose(w_tmp, w.grad.data):
            #     print('dconv!!!')
            #     sys.exit(0)
            # print('is equal 1:', torch.allclose(w_tmp, w.grad.data))
        else:
            # w_tmp = w.grad.data.clone()
            w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(p.data)).view_as(w)
            # if not torch.allclose(w_tmp, w.grad.data):
            #     print('conv!!!')
            #     sys.exit(0)
            # print('is equal 2:', torch.allclose(w_tmp, w.grad.data))
        return p

    def train(self, data_loader):
        train_losses = []
        total_iters = 0
        start_time = time.time()
        self.clipgrad = 10
        # self.optimizer = self._get_optimizer()
        for epoch in range(self.num_epochs):
            if epoch < 80:
                immune = False
            else:
                immune = True
            prev_time = datetime.now()
            self.REDCNN.train(True)
            count = 0
            for iter_, (x, y) in enumerate(data_loader):
                count += 1
                length = y.shape[0]
                lamda = iter_ / length / self.num_epochs + epoch / self.num_epochs
                total_iters += 1
                # add 1 channel
                x = x.unsqueeze(1).float().to(self.device)
                y = y.unsqueeze(1).float().to(self.device)
                alpha_array = [1.0 * 0.00001 ** lamda, 1.0 * 0.0001 ** lamda, 1.0 * 0.001 ** lamda, 1.0 * 0.01 ** lamda,
                               1.0 * 0.5 ** lamda, 0.6]

                def update_weight(x_list, immune=True):
                    for n, w in self.REDCNN.named_parameters():
                        if n == 'module.conv_first.weight':
                            self.P_conv1 = self.pro_weight(self.P_conv1, x_list[0], w, alpha=alpha_array[0],
                                                           immune=immune, stride=1, kernel_size=5)
                        if n == 'module.conv2.weight':
                            self.P_conv2 = self.pro_weight(self.P_conv2, x_list[1], w, alpha=alpha_array[1],
                                                           immune=immune, stride=1, kernel_size=5)
                        if n == 'module.conv3.weight':
                            self.P_conv3 = self.pro_weight(self.P_conv3, x_list[2], w, alpha=alpha_array[2],
                                                           immune=immune, stride=1, kernel_size=5)
                        if n == 'module.conv4.weight':
                            self.P_conv4 = self.pro_weight(self.P_conv4, x_list[3], w, alpha=alpha_array[3],
                                                           immune=immune, stride=1, kernel_size=5)
                        if n == 'module.conv5.weight':
                            self.P_conv5 = self.pro_weight(self.P_conv5, x_list[4], w, alpha=alpha_array[4],
                                                           immune=immune, stride=1, kernel_size=5)
                        if n == 'module.conv_1.weight':
                            self.P_dconv1 = self.pro_weight(self.P_dconv1, x_list[5], w, alpha=alpha_array[4],
                                                            immune=immune, dconv=True, stride=1, kernel_size=5)
                        if n == 'module.conv_2.weight':
                            self.P_dconv2 = self.pro_weight(self.P_dconv2, x_list[6], w, alpha=alpha_array[4],
                                                            immune=immune, dconv=True, stride=1, kernel_size=5)
                        if n == 'module.conv_3.weight':
                            self.P_dconv3 = self.pro_weight(self.P_dconv3, x_list[7], w, alpha=alpha_array[4],
                                                            immune=immune, dconv=True, stride=1, kernel_size=5)
                        if n == 'module.conv_4.weight':
                            self.P_dconv4 = self.pro_weight(self.P_dconv4, x_list[8], w, alpha=alpha_array[4],
                                                            immune=immune, dconv=True, stride=1, kernel_size=5)
                        if n == 'module.conv_t_last.weight':
                            self.P_dconv5 = self.pro_weight(self.P_dconv5, x_list[9], w, alpha=alpha_array[5],
                                                            immune=immune, dconv=True, stride=1, kernel_size=5)

                x_list, pred = self.REDCNN(x)
                loss = self.criterion(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                update_weight(x_list, immune=immune)
                if immune:
                    torch.nn.utils.clip_grad_norm_(self.REDCNN.parameters(), self.clipgrad)

                train_losses.append(loss.item())
                self.optimizer.step()
                if total_iters % self.print_iters == 0:
                    Logger.append("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s"
                                  .format(total_iters, epoch, self.num_epochs, iter_ + 1, len(data_loader),
                                          loss.item(), time.time() - start_time))
                # learning rate decay
                self.lr_decay(epoch)
                #if total_iters % self.decay_iters == 0:
                #    self.lr_decay(epoch)

                # save model
                # if total_iters % self.save_iters == 0:
                #     self.save_model(total_iters)
                    # np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))
            self.save_model(epoch+1)
            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "Time %02d:%02d:%02d" % (h, m, s)
            Logger.append(time_str)
        np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))

    def test(self, data_loader):
        self.REDCNN.to(self.device)
        self.load_model(self.test_iters)
        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = [], [], []
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = [], [], []

        with torch.no_grad():
            for i, (x, y) in enumerate(data_loader):
                shape_ = x.shape[-1]
                x = x.unsqueeze(1).float().to(self.device)
                y = y.unsqueeze(1).float().to(self.device)
                _, pred = self.REDCNN(x)
                x = x.view(shape_, shape_).cpu().detach()
                y = y.view(shape_, shape_).cpu().detach()
                pred = pred.view(shape_, shape_).cpu().detach()

                original_ssim = measure.compare_ssim(x.numpy(), y.numpy(), data_range=1.0)
                original_psnr = measure.compare_psnr(x.numpy(), y.numpy(), data_range=1.0)
                original_rmse = measure.compare_mse(x.numpy(), y.numpy())
                original_rmse = np.sqrt(original_rmse)
                ori_psnr_avg.append(original_psnr)
                ori_ssim_avg.append(original_ssim)
                ori_rmse_avg.append(original_rmse)

                pred_ssim = measure.compare_ssim(y.numpy(), pred.numpy(), data_range=1.0)
                pred_psnr = measure.compare_psnr(y.numpy(), pred.numpy(), data_range=1.0)
                pred_rmse = measure.compare_mse(y.numpy(), pred.numpy())
                pred_rmse = np.sqrt(pred_rmse)
                pred_psnr_avg.append(pred_psnr)
                pred_ssim_avg.append(pred_ssim)
                pred_rmse_avg.append(pred_rmse)

                # x = self.trunc(self.denormalize_(x))
                # y = self.trunc(self.denormalize_(y))
                # pred = self.trunc(self.denormalize_(pred))
                # save every image
                #self.save_raw(x, y, pred, i)
                # self.save_fig(x, y, pred, i, [original_psnr, original_ssim, original_rmse], [pred_psnr, pred_ssim, pred_rmse])

                # save result figure
                # if i % 30 == 0:
                #    self.save_fig(x, y, pred, i, [original_psnr, original_ssim, original_rmse], [pred_psnr, pred_ssim, pred_rmse])
                # self.save_raw(x, y, pred, i)
            ori_psnr_avg = np.array(ori_psnr_avg, dtype=np.float32)
            ori_rmse_avg = np.array(ori_rmse_avg, dtype=np.float32)
            ori_ssim_avg = np.array(ori_ssim_avg, dtype=np.float32)
            pred_psnr_avg = np.array(pred_psnr_avg, dtype=np.float32)
            pred_rmse_avg = np.array(pred_rmse_avg, dtype=np.float32)
            pred_ssim_avg = np.array(pred_ssim_avg, dtype=np.float32)
            ori_psnr_mean, ori_psnr_std = np.mean(ori_psnr_avg), np.std(ori_psnr_avg)
            ori_rmse_mena, ori_rmse_std = np.mean(ori_rmse_avg), np.std(ori_rmse_avg)
            ori_ssim_mean, ori_ssim_std = np.mean(ori_ssim_avg), np.std(ori_ssim_avg)
            pred_psnr_mean, pred_psnr_std = np.mean(pred_psnr_avg), np.std(pred_psnr_avg)
            pred_rmse_mena, pred_rmse_std = np.mean(pred_rmse_avg), np.std(pred_rmse_avg)
            pred_ssim_mean, pred_ssim_std = np.mean(pred_ssim_avg), np.std(pred_ssim_avg)
            Logger.append('Original\nPSNR mean: {:.4f},PSNR std: {:.4f} \nRMSE mean: {:.4f}, RMSE std: {:.4f}'
                          '\nSSIM mean: {:.4f}, SSIM std: {:.4f}'.format(ori_psnr_mean,
                                                                         ori_psnr_std, ori_rmse_mena, ori_rmse_std,
                                                                         ori_ssim_mean, ori_ssim_std))
            Logger.append('After learning\nPSNR mean: {:.4f},PSNR std: {:.4f} \nRMSE mean: {:.4f}, RMSE std: {:.4f}'
                          '\nSSIM mean: {:.4f}, SSIM std: {:.4f}'.format(pred_psnr_mean,
                                                                         pred_psnr_std, pred_rmse_mena, pred_rmse_std,
                                                                         pred_ssim_mean, pred_ssim_std))



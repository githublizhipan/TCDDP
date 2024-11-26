import glob
import os, utils
import time
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from models import TCDDP
import random
from Evaluation.Eval_metrics import compute_surface_distances, compute_average_surface_distance, \
    compute_robust_hausdorff


def csv_writter(line, name):
    with open(name + '.csv', 'a') as file:
        file.write(line)
        file.write('\n')


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


same_seeds(24)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)


def make_one_hot(mask, num_class):
    mask_unique = [m for m in range(num_class)]
    one_hot_mask = [mask == i for i in mask_unique]
    one_hot_mask = np.stack(one_hot_mask)
    return one_hot_mask


def main():
    num_class = 57  # 57个ROI
    val_dir = 'LPBA_path/Val/'
    spacing = (1, 1, 1)
    model_idx = -1
    model_folder = 'model_save_dir/'
    model_dir = 'experiments/' + model_folder
    save_root = 'experiments/Results'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    dict = utils.get_LPBAlabel()
    if os.path.exists('experiments/' + model_folder[:-1] + '.csv'):
        os.remove('experiments/' + model_folder[:-1] + '.csv')
    csv_writter(model_folder[:-1], 'experiments/' + model_folder[:-1])
    line = ''
    for i in range(54):
        line = line + ',' + dict[0][i]
    csv_writter(line, 'experiments/' + model_folder[:-1])

    img_size = (160, 192, 160)
    model = TCDDP(img_size)

    parameters = count_parameters(model)  # 统计模型参数量
    print("parameters: ", parameters)

    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.LPBABrainInferDatasetS2S(glob.glob(val_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    # DSC
    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    # ASSD
    eval_assd_def = utils.AverageMeter()
    eval_assd_raw = utils.AverageMeter()
    # HD95
    eval_hd95_def = utils.AverageMeter()
    eval_hd95_raw = utils.AverageMeter()
    # Jacobian
    eval_det = AverageMeter()
    # time
    eval_time = utils.AverageMeter()

    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            save_name = data[4][0]
            data = [t.cuda() for t in data[:4]]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            time_start = time.time()

            x_def, flow = model(x, y)
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])  # 变形后的x_seg

            time_end = time.time()
            eval_time.update((time_end - time_start))

            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            # 计算jacobian
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])

            line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            line = line  # +','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            csv_writter(line, 'experiments/' + model_folder[:-1])

            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

            # 计算DSC
            dsc_trans = utils.dice_val_VOI(def_out.long(), y_seg.long())
            dsc_raw = utils.dice_val_VOI(x_seg.long(), y_seg.long())
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))

            seg_trans = make_one_hot(def_out.detach().cpu().numpy()[0, 0, :, :, :], num_class=num_class)
            seg_x = make_one_hot(x_seg.detach().cpu().numpy()[0, 0, :, :, :], num_class=num_class)
            seg_y = make_one_hot(y_seg.detach().cpu().numpy()[0, 0, :, :, :], num_class=num_class)
            assd_trans = 0
            assd_raw = 0
            hd95_trans = 0
            hd95_raw = 0
            cal_index = 0

            for i in range(seg_trans.shape[0]):
                if i == 0:
                    continue

                if (seg_trans[i] == False).all() or (seg_y[i] == False).all() or (seg_x[i] == False).all():
                    continue
                sur_dist_trans = compute_surface_distances(seg_trans[i], seg_y[i], spacing_mm=spacing)
                sur_dist_raw = compute_surface_distances(seg_x[i], seg_y[i], spacing_mm=spacing)

                a, b = compute_average_surface_distance(sur_dist_trans)
                assd_trans += (a + b) / 2
                c, d = compute_average_surface_distance(sur_dist_raw)
                assd_raw += (c + d) / 2
                hd95_trans += compute_robust_hausdorff(sur_dist_trans, 95)
                hd95_raw += compute_robust_hausdorff(sur_dist_raw, 95)
                cal_index += 1
            # ASSD
            assd_trans /= cal_index
            assd_raw /= cal_index
            # HD95
            hd95_trans /= cal_index
            hd95_raw /= cal_index
            print('Trans assd: {:.4f}, Raw assd: {:.4f}'.format(assd_trans.item(), assd_raw.item()))
            print('Trans hd95: {:.4f}, Raw hd95: {:.4f}'.format(hd95_trans.item(), hd95_raw.item()))
            eval_assd_def.update(assd_trans.item(), x.size(0))
            eval_assd_raw.update(assd_raw.item(), x.size(0))
            eval_hd95_def.update(hd95_trans.item(), x.size(0))
            eval_hd95_raw.update(hd95_raw.item(), x.size(0))

            stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('Deformed ASSD: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_assd_def.avg,
                                                                                     eval_assd_def.std,
                                                                                     eval_assd_raw.avg,
                                                                                     eval_assd_raw.std))
        print('Deformed HD95: {:.3f} +- {:.3f}, Affine hd95: {:.3f} +- {:.3f}'.format(eval_hd95_def.avg,
                                                                                      eval_hd95_def.std,
                                                                                      eval_hd95_raw.avg,
                                                                                      eval_hd95_raw.std))

        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))
        print("time: {}", eval_time.avg)


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()

#!/usr/bin/env python
import os
import pathlib
import math

import matplotlib.pyplot as plt

import numpy as np
import torch
import tqdm
from mpiifacegaze_eval_lib.utils.utils import AverageMeter
from mpiifacegaze_eval_lib.opts import opts
from mpiifacegaze_eval_lib.datasets.dataset_factory import get_dataset
from mpiifacegaze_eval_lib.models.model import create_model, load_model, save_model
from mpiifacegaze_eval_lib.models.decode import ctdet_gaze_decode
from mpiifacegaze_eval_lib.utils.post_process import ctdet_gaze_post_process


def euclidean_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def L2_distance(pred, gt):

    x1, y1 = pred
    x2, y2 = gt
    error = euclidean_distance(x1, y1, x2, y2)
    
    return error


def test(model, test_loader, opt):
    model.eval()
    torch.cuda.empty_cache()
    device = torch.device(opt.device)
    
    model = model.to(device)
    
    L2_errors = AverageMeter()
    losses_gaze = AverageMeter()

    # predictions = []
    # gts = []
    with torch.no_grad():
        for iter_id, batch in enumerate(test_loader):
            
            
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True) 
            output = model(batch['input'])
            # print(output)
            output = output[-1]
            reg = output['reg'] if opt.reg_offset else None
            
            dets = ctdet_gaze_decode(output['hm'], reg=reg, K=opt.K)
            # print(f"dets: {dets}")
            dets = dets.detach().cpu().numpy()
            dets = dets.reshape(1, -1, dets.shape[2])
            dets[:, :, :2] *= opt.down_ratio
            # print(dets)
            dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
            # print(dets_gt)
            # dets_gt[:, :, :2] *= opt.down_ratio
            # print(dets_gt)
            # print(dets_gt.shape)
            # for i in range(1):

            #     img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            #     vp_img = batch['vp'][i].detach().cpu().numpy().transpose(1, 2, 0)
            #     img = np.clip(((
            #         img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            # print(output.shape)
            dets_out = ctdet_gaze_post_process(
                dets.copy(), batch['meta']['vp_c'].cpu().numpy(),
                batch['meta']['vp_s'].cpu().numpy(),
                output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])

            # print(f"dets_out: {dets_out}")
            for i in range(1):
                cls_id = 1
                print(f"dets_out: {dets_out[i][cls_id][0:1]}")
                dets_coord = dets_out[i][cls_id][0][:2]
            
            dets_gt_out = ctdet_gaze_post_process(
                dets_gt.copy(), batch['meta']['vp_c'].cpu().numpy(),
                batch['meta']['vp_s'].cpu().numpy(),
                batch['hm'].shape[2], batch['hm'].shape[3], batch['hm'].shape[1])

            for i in range(1):
                cls_id = 1
                # print(f"dets_gt_out: {dets_gt_out[i][cls_id][0][:2]}")
                dets_gt_coord = dets_gt_out[i][cls_id][0][:2]
                print(f"dets_gt_coord: {dets_gt_coord}")
                
            vp_gazepoint = batch['meta']['vp_gazepoint'].cpu().numpy()
            print(f"vp_gazepoint : {vp_gazepoint}")
            

            output_hm = output["hm"][0].cpu()
            # print(f"output_hm: {output_hm}")
            output_hm = output_hm.permute(1, 2, 0)
            max_index = torch.argmax(output_hm)
            max_coord = torch.nonzero(max_index.view(-1) == torch.arange(output_hm.numel()), as_tuple=False)
            max_coord_2d = (max_coord % output_hm.size(1), max_coord // output_hm.size(1))
            print(f"output_hm max index: {max_coord_2d}")
            
            batch_hm = batch["hm"][0].cpu()
            # batch_hm = batch_hm.transpose(1,2,0)
            batch_hm = batch_hm.permute(1, 2, 0)

            plt.imshow(output_hm, cmap="gray")
            plt.axis('off')  # 关闭坐标轴
            plt.show()
            
            plt.imshow(batch_hm, cmap="gray")
            plt.axis('off')  # 关闭坐标轴
            plt.show()
            
            
            L2_error = L2_distance(dets_coord, dets_gt_coord)
            L2_errors.update(L2_error)
            break

    mean_L2_errors = L2_errors.avg

            
    return mean_L2_errors


def main(opt):
    print("mpiifacegaze eval")
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/model_best.pth"
    

    model = load_model(model, model_path)
    # if opt.load_model != '':
    #     model, optimizer, start_epoch = load_model(
    #     model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    # Trainer = train_factory[opt.task]
    # trainer = Trainer(opt, model, optimizer)
    # trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    

    
    test_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
    )
    
    

    
    # config = load_config()

    # output_rootdir = pathlib.Path(config.test.output_dir)
    # checkpoint_name = pathlib.Path(config.test.checkpoint).stem
    # output_dir = output_rootdir / checkpoint_name
    # output_dir.mkdir(exist_ok=True, parents=True)
    # save_config(config, output_dir)

    # test_loader = create_dataloader(config, is_train=False)

    # model = create_model(config)
    # checkpoint = torch.load(config.test.checkpoint, map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    
    test(model, test_loader, opt)

    L2_distance = test(model, test_loader, opt)
    # print(L2_distance.shape)

    print(f'The mean error distance (mm): {L2_distance:.2f}')

    # output_path = output_dir / 'predictions.npy'
    # np.save(output_path, predictions.numpy())
    # output_path = output_dir / 'gts.npy'
    # np.save(output_path, gts.numpy())
    # output_path = output_dir / 'error.txt'
    # with open(output_path, 'w') as f:
    #     f.write(f'{angle_error}')


if __name__ == '__main__':
  opt = opts().parse()
  main(opt)

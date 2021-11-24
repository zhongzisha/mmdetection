# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings
import pickle
import shutil
import tempfile
import numpy as np

import cv2
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import typing
import zlib
import base64
import cupy as cp
import gc


# for Kaggle submission
# import os
# os.environ['CUDA_PATH'] = '/usr/local/cuda'
# os.environ['CUDA_HOME'] = '/usr/local/cuda'
# os.environ['MMCV_WITH_OPS'] = '1'
# os.environ['TORCH_CUDA_ARCH_LIST'] = '6.0'
#
# !pip install ../input/packages/addict-2.4.0-py3-none-any.whl
# !pip install ../input/packages/yapf-0.31.0-py2.py3-none-any.whl
# !pip install ../input/packages/pycocotools-2.0.2/dist/pycocotools-2.0.2.tar
# !pip install ../input/packages/mmcv_mmdet/mmcv-1.3.10
# !pip install ../input/packagesv2/terminaltables-3.1.0
# !pip install ../input/packages/mmcv_mmdet/mmdetection-2.17.0


def encode_binary_mask(mask: np.ndarray) -> typing.Text:
    """Converts a binary mask into OID challenge encoding ascii text."""
    # check input mask --
    if mask.dtype != np.bool:
        raise ValueError(
            "encode_binary_mask expects a binary mask, received dtype == %s" %
            mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask expects a 2d mask, received shape == %s" %
            mask.shape)

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = cocomask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode()


def mask2rle(msk):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    msk = cp.array(msk)
    pixels = msk.flatten()
    pad = cp.array([0])
    pixels = cp.concatenate([pad, pixels, pad])
    runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) >= 6:
        return ' '.join(str(x) for x in runs)
    else:
        return None


def rle2mask(rle, shape=None):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    if shape is None:
        shape = [520, 704]
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


def one_hot(y, num_classes, dtype=cp.uint8):  # GPU
    y = cp.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = cp.max(y) + 1
    n = y.shape[0]
    categorical = cp.zeros((n, num_classes), dtype=dtype)
    categorical[cp.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = cp.reshape(categorical, output_shape)
    return categorical


def fix_overlap(msk):  # GPU
    """
    Args:
        mask: multi-channel mask, each channel is an instance of cell, shape:(520,704,None)
    Returns:
        multi-channel mask with non-overlapping values, shape:(520,704,None)
    """
    msk = cp.array(msk)
    msk = cp.pad(msk, [[0, 0], [0, 0], [1, 0]])  # add dummy mask for background
    ins_len = msk.shape[-1]
    msk = cp.argmax(msk, axis=-1)  # convert multi channel mask to single channel mask, argmax will remove overlap
    msk = one_hot(msk, num_classes=ins_len)  # back to multi-channel mask, some instance might get removed
    msk = msk[..., 1:]  # remove background mask
    msk = msk[..., cp.any(msk, axis=(0, 1))]  # remove all-zero masks
    # assert np.prod(msk, axis=-1).sum()==0 # overlap check, will raise error if there is overlap
    return msk


def check_overlap(msk):
    msk = msk.astype(cp.bool).astype(cp.uint8)  # binary mask
    return cp.any(cp.sum(msk, axis=-1) > 1)  # only one channgel will contain value


# From https://www.kaggle.com/stainsby/fast-tested-rle
def rle_decode(mask_rle, shape=(520, 704)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) >= 6:
        return ' '.join(str(x) for x in runs)
    else:
        return None


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    json_filename=None,
                    img_postfix='.jpg'):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    idx = 0
    if os.path.exists(json_filename):
        coco = COCO(json_filename)
        img_maps = {v['file_name'].replace(img_postfix, ''): k for k, v in coco.imgs.items()}
    else:
        coco = None
        img_maps = None
    all_results = []
    alldata = []
    for _, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        # import pdb
        # pdb.set_trace()

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                img_prefix = os.path.basename(img_meta['ori_filename']).replace(img_postfix, '')
                if img_maps:
                    # import pdb
                    # pdb.set_trace()
                    img_id = img_maps[img_prefix]
                    ann_ids = coco.get_ann_ids(img_ids=[img_id])
                    anns = coco.load_anns(ann_ids)
                    gt_bboxes = []
                    gt_labels = []
                    for ann in anns:
                        x1, y1, bw, bh = ann['bbox']
                        gt_bboxes.append([x1, y1, x1 + bw, y1 + bh])
                        gt_labels.append(ann['category_id'])
                    gt_bboxes = np.array(gt_bboxes)
                    gt_labels = np.array(gt_labels)
                    # gt_bboxes = dataset.get_ann_info(idx)['bboxes']
                    # gt_labels = dataset.get_ann_info(idx)['labels']
                    idx += 1
                    for gt_box, gt_label in zip(gt_bboxes, gt_labels):
                        gt_box = gt_box.astype(np.int32)
                        cv2.rectangle(img_show, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]),
                                      color=(0, 255, 0))
                        cv2.putText(img_show,
                                    text='%s: %d' % (img_meta['ori_filename'], len(gt_bboxes)),
                                    org=(50, 100),
                                    fontScale=2,
                                    fontFace=1,
                                    color=(255, 0, 0))

                img_show2 = np.copy(img_show)

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

                encoded_masks = []
                print(len(result[i][0]))

                # import pdb
                # pdb.set_trace()

                mask = []
                pred_masks = []
                for class_id in range(3):
                    print('class_id', class_id)
                    bbs = result[i][0][class_id]
                    sgs = result[i][1][class_id]
                    # print(bbs, sgs, type(bbs), type(sgs))
                    for bb, sg in zip(bbs, sgs):
                        box = bb[:4]
                        cnf = bb[4]
                        mask.append(cp.array(sg))
                        pred_masks.append(sg)
                        if out_dir:
                            cv2.rectangle(img_show2,
                                          (int(box[0]), int(box[1])),
                                          (int(box[2]), int(box[3])),
                                          color=(0, 255, 255))
                            cv2.putText(img_show2,
                                        text='%.2f' % cnf,
                                        org=(int(box[0]), int(box[1])),
                                        fontScale=1,
                                        fontFace=1,
                                        color=(255, 0, 0))
                if len(mask) == 0:
                    continue
                mask = cp.stack(mask, axis=-1)
                if check_overlap(mask):  # if mask instances have overlap then fix it
                    mask = fix_overlap(mask)
                for idx in range(mask.shape[-1]):
                    mask_ins = mask[..., idx]
                    rle = mask2rle(mask_ins)
                    if rle:
                        all_results.append([img_prefix, rle])

                used = np.zeros((ori_h, ori_w), dtype=int)
                for mask1 in pred_masks:
                    mask1 = mask1 * (1 - used)
                    used += mask1
                    rle = rle_encode(mask1)
                    if rle:
                        alldata.append([img_prefix, rle])
                        encoded_masks.append(rle)

                mask_all = None
                if out_dir:
                    mask_all = cp.sum(mask, axis=-1) > 0
                    print(type(mask_all), mask_all.shape)
                    mask_all = mask_all.get()

                    cv2.imwrite(out_file.replace(img_postfix, '_predicted' + img_postfix),
                                np.concatenate([
                                    img_show2,
                                    np.stack([mask_all, mask_all, mask_all], axis=-1) * 255
                                ], axis=1)
                                )

                del mask, rle, sgs, bbs, mask_all, pred_masks
                gc.collect()

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()

    # print(all_results)
    # import pdb
    # pdb.set_trace()

    if len(all_results) > 0:
        import pandas as pd
        pred_df = pd.DataFrame(all_results, columns=['id', 'predicted'])
        sub_df = pd.read_csv('configs/cell_seg/sample_submission.csv')
        del sub_df['predicted']
        sub_df = sub_df.merge(pred_df, on='id', how='left')
        sub_df.to_csv(os.path.join(out_dir, 'submission.csv'), index=False)
        sub_df.head()

        # pdb.set_trace()

    # if len(alldata) > 0:
    #     import pandas as pd
    #     pred_df = pd.DataFrame(alldata, columns=['id', 'predicted'])
    #     sub_df = pd.read_csv('configs/cell_seg/sample_submission.csv')
    #     del sub_df['predicted']
    #     sub_df = sub_df.merge(pred_df, on='id', how='left')
    #     sub_df.to_csv('submission1.csv', index=False)
    #     sub_df.head()

    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
             ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--json_filename', default='', type=str)
    parser.add_argument(
        '--img_postfix', default='.jpg', type=str)
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function (deprecate), '
             'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr,
                                  args.json_filename,
                                  args.img_postfix)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main()

from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import glob,os
import time
import random
import copy
import mmcv
import numpy as np
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import cv2
from osgeo import gdal, osr
from natsort import natsorted
from pathlib import Path
import json
import psutil
from yoloV5.myutils import load_gt_from_txt, load_gt_from_esri_xml, py_cpu_nms, \
    compute_offsets, save_predictions_to_envi_xml, LoadImages, \
    box_iou, ap_per_class, ConfusionMatrix
from yoloV5.utils.torch_utils import select_device, time_synchronized


def load_classifier(name='resnet101', n=2):
    # Loads a pretrained model reshaped to n-class output
    model = torchvision.models.__dict__[name](pretrained=True)

    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def apply_classifier(x, model, img, im0, ti=0, oi=0, save_root='./'):
    # applies a second stage classifier to yolo outputs

    dtype = x[0].dtype
    device = x[0].device
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=dtype, device=device)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=dtype, device=device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    save_dir = '%s/cls_results/%d_%d' % (save_root, ti, oi)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    height, width = im0.shape[:2]
    newx = []
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                if pred_cls1[j] == 2:  # 0:small, 1:mid, 2:large, 3:jueyuanzi
                    # print(a)
                    xmin, ymin, xmax, ymax = a[:4]
                    xmin = int(max(0, xmin))
                    ymin = int(max(0, ymin))
                    xmax = int(min(width - 1, xmax))
                    ymax = int(min(height - 1, ymax))
                    cutout = im0[ymin:ymax, xmin:xmax]
                    im = cv2.resize(cutout, (224, 224))  # BGR
                    # cv2.imwrite('test%i.jpg' % j, cutout)

                    im = im.transpose(2, 0, 1)  # RGB, to 3x416x416
                    im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                    im /= 255.0  # 0 - 255 to 0.0 - 1.0
                    ims.append(im)
            if len(ims) > 0:
                ims_copy = copy.deepcopy(ims)
                ims = torch.Tensor(ims).to(d.device)
                ims.sub_(mean).div_(std)
                logits = model(ims)
                prob_op = nn.Softmax(dim=1)
                probs = prob_op(logits)
                pred_0 = logits.argmax(1)  # classifier prediction
                pred_0 = 1 - pred_0  # 0 is pos, 1 is neg
                print('probs', probs)

            ii = 0
            newd = []
            for j, a in enumerate(d):  # per item
                if pred_cls1[j] == 2:
                    if pred_0[ii] == 0:
                        newd.append(x[i][j])
                    # if probs[ii, 1] > 0.001:  # TODO 阈值需要设定，这里用阈值是因为发现有的正样本分成了负样本
                    #     newd.append(x[i][j])
                    else:
                        # 这里保存那些检测为杆塔但是分类不是杆塔的图片
                        cv2.imwrite('%s/%d_%f_%f.jpg' % (save_dir, j, d[j, 4], probs[ii, 1]),
                                    (ims_copy[ii]*255).astype(np.uint8).transpose([1,2,0]))
                        pass
                    ii += 1
                else:
                    newd.append(x[i][j])
            # newd = x[i][pred_cls1 == pred_cls2]  # retain matching class detections
            if len(newd) > 0:
                newd = torch.stack(newd)
                newx.append(newd)

    return newx


def main():
    parser = ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--cls-weights', type=str, default='', help='cls_model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--gt-xml-dir', type=str, default='', help='gt xml dir')
    parser.add_argument('--gt-prefix', type=str, default='', help='gt prefix')
    parser.add_argument('--gt-subsize', type=int, default=5120, help='train image size for labeling')
    parser.add_argument('--gt-gap', type=int, default=128, help='train gap size for labeling')
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--big-subsize', type=int, default=51200, help='inference big-subsize (pixels)')
    parser.add_argument('--gap', type=int, default=128, help='overlap size')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size')

    parser.add_argument('--score-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--hw-thres', type=float, default=5, help='height or width threshold for box')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    args = parser.parse_args()

    source, weights, view_img, save_txt, imgsz, gap, \
    gt_xml_dir, gt_prefix, gt_subsize, gt_gap, \
    big_subsize, batchsize, score_thr, hw_thr = \
        args.source, args.weights, args.view_img, args.save_txt, args.img_size, args.gap, \
        args.gt_xml_dir, args.gt_prefix, int(args.gt_subsize), int(args.gt_gap), args.big_subsize, \
        args.batchsize, args.score_thres, args.hw_thres

    # Directories
    save_dir = Path(args.project)  # increment run
    if not os.path.exists(save_dir):
        save_dir.mkdir(parents=True, exist_ok=True)

    names = {0: '1', 1: '2', 2: '3', 3: '4'}
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    shown_labels = [0, 1, 2, 3]  # 只显示中大型杆塔和绝缘子

    device = select_device(args.device)

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.weights, device=device)
    stride = 32

    # Second-stage classifier
    classify = False
    if args.cls_weights != '':
        classify = True
    if classify:
        print(args.cls_weights)
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load(r'%s' % args.cls_weights, map_location=device))
        modelc = modelc.to(device)
        modelc.eval()

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    seen = 0
    nc = len(names)
    confusion_matrix = ConfusionMatrix(nc=nc)
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    iouv = torch.linspace(0.1, 0.95, 20).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    is_plot = True

    gt_json_dict = {}
    gt_json_dict['images'] = []
    gt_json_dict['categories'] = []
    gt_json_dict['annotations'] = []
    for index, name in enumerate(names):  # 1 is gan, 2 is jueyuanzi
        single_cat = {'id': index + 1, 'name': name, 'supercategory': name}
        gt_json_dict['categories'].append(single_cat)
    jdict = []

    inst_count = 1

    for ti in range(len(tiffiles)):
        image_id = ti + 1
        tiffile = tiffiles[ti]
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')

        print(ti, '=' * 80)
        print(file_prefix)

        ds = gdal.Open(tiffile, gdal.GA_ReadOnly)
        print("Driver: {}/{}".format(ds.GetDriver().ShortName,
                                     ds.GetDriver().LongName))
        print("Size is {} x {} x {}".format(ds.RasterXSize,
                                            ds.RasterYSize,
                                            ds.RasterCount))
        print("Projection is {}".format(ds.GetProjection()))
        projection = ds.GetProjection()
        projection_sr = osr.SpatialReference(wkt=projection)
        projection_esri = projection_sr.ExportToWkt(["FORMAT=WKT1_ESRI"])
        geotransform = ds.GetGeoTransform()
        xOrigin = geotransform[0]
        yOrigin = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        orig_height, orig_width = ds.RasterYSize, ds.RasterXSize
        if geotransform:
            print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
            print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
            print("IsNorth = ({}, {})".format(geotransform[2], geotransform[4]))

        save_path = str(save_dir) + '/' + file_prefix + '_uncompressed.tif'
        # tifffile.imwrite(save_path, im0)

        # 画gt
        if gt_xml_dir != '' and os.path.exists(gt_xml_dir):
            # 加载gt，分两部分，一部分是txt格式的。一部分是esri xml格式的
            gt_boxes1, gt_labels1 = load_gt_from_txt(os.path.join(gt_xml_dir, file_prefix + '_gt.txt'))
            gt_boxes2, gt_labels2 = load_gt_from_esri_xml(os.path.join(gt_xml_dir, file_prefix + '_gt_5.xml'),
                                                          gdal_trans_info=geotransform)
            gt_boxes = gt_boxes1 + gt_boxes2
            gt_labels = gt_labels1 + gt_labels2

            if len(gt_boxes) > 0:
                all_boxes = np.concatenate([np.array(gt_boxes, dtype=np.float32).reshape(-1, 4),
                                            np.array(gt_labels, dtype=np.float32).reshape(-1, 1)], axis=1)
                print('all_boxes')
                print(all_boxes)

                # 每个类进行nms
                tmp_boxes = []
                tmp_labels = []
                for label in [1, 2, 3, 4]:
                    idx = np.where(all_boxes[:, 4] == label)[0]
                    if len(idx) > 0:
                        boxes_thisclass = all_boxes[idx, :4]
                        labels_thisclass = all_boxes[idx, 4]
                        dets = np.concatenate([boxes_thisclass.astype(np.float32),
                                               0.99 * np.ones_like(idx, dtype=np.float32).reshape([-1, 1])], axis=1)
                        keep = py_cpu_nms(dets, thresh=0.5)
                        tmp_boxes.append(boxes_thisclass[keep])
                        tmp_labels.append(labels_thisclass[keep])
                gt_boxes = np.concatenate(tmp_boxes)
                gt_labels = np.concatenate(tmp_labels)

                # # 把那些太小的去掉试试
                # gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
                # gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
                # gt_l = gt_labels.copy()
                #
                # print('len(gt_boxes): ', len(gt_boxes))
                # print('len(gt_labels): ', len(gt_labels))
                #
                # inds = np.where((gt_l == 1) & (gt_w <= 100) & (gt_h <= 100))[0]
                # gt_boxes = np.delete(gt_boxes, inds, axis=0)
                # gt_labels = np.delete(gt_labels.reshape([-1, 1]), inds, axis=0).reshape([-1])
                #
                # print('len(gt_boxes) after: ', len(gt_boxes))
                # print('len(gt_labels) after: ', len(gt_labels))
                # import pdb
                # pdb.set_trace()

                gt_boxes = torch.from_numpy(gt_boxes).to(device)
                gt_labels = torch.from_numpy(gt_labels).to(device)
            else:
                gt_boxes = []
                gt_labels = []
        else:
            gt_boxes = []
            gt_labels = []

        if len(gt_boxes) == 0:
            import pdb
            pdb.set_trace()

        # 先计算可用内存，如果可以放得下，就不用分块了
        avaialble_mem_bytes = psutil.virtual_memory().available
        if False:# orig_width * orig_height * ds.RasterCount < 0.8 * avaialble_mem_bytes:
            offsets = [[0, 0, orig_width, orig_height]]
        else:
            # 根据big_subsize计算子块的起始偏移
            offsets = compute_offsets(height=orig_height, width=orig_width, subsize=big_subsize, gap=2 * gt_gap)
        print('offsets: ', offsets)

        all_preds_filename = str(save_dir) + '/' + file_prefix + '_all_preds.pt'
        if os.path.exists(all_preds_filename):
            all_preds = torch.load(all_preds_filename)
        else:
            if view_img:
                driver = gdal.GetDriverByName("GTiff")
                outdata = driver.Create(save_path, orig_width, orig_height, 3, gdal.GDT_Byte)
                # options=['COMPRESS=LZW', 'BIGTIFF=YES', 'INTERLEAVE=PIXEL'])
                # outdata = driver.CreateCopy(save_path, ds, 0, ['COMPRESS=LZW', 'BIGTIFF=YES', 'INTERLEAVE=PIXEL'])
                outdata.SetGeoTransform(geotransform)  # sets same geotransform as input
                outdata.SetProjection(projection)  # sets same projection as input

            all_preds = []
            for oi, (xoffset, yoffset, sub_width, sub_height) in enumerate(offsets):  # left, up
                # sub_width = min(orig_width, big_subsize)
                # sub_height = min(orig_height, big_subsize)
                # if xoffset + sub_width > orig_width:
                #     sub_width = orig_width - xoffset
                # if yoffset + sub_height > orig_height:
                #     sub_height = orig_height - yoffset

                print('processing sub image %d' % oi, xoffset, yoffset, sub_width, sub_height)
                dataset = LoadImages(gdal_ds=ds, xoffset=xoffset, yoffset=yoffset,
                                     width=sub_width, height=sub_height,
                                     batchsize=batchsize, subsize=imgsz, gap=gap, stride=stride,
                                     return_list=True)
                if len(dataset) == 0:
                    continue

                print('forward inference')
                sub_preds = []
                for img in dataset:


                    # cv2.imwrite('test1.png', img[0])
                    # cv2.imwrite('test2.png', img[1])

                    # print(len(img))
                    # import pdb
                    # pdb.set_trace()

                    result = inference_detector(model, img)

                    if isinstance(result, tuple):
                        bbox_results, segm_results = result
                    else:
                        bbox_results, segm_results = result, None
                    #
                    # import pdb
                    # pdb.set_trace()

                    pred_per_image = []
                    for bbox_result in bbox_results:

                        bboxes = np.concatenate(bbox_result, axis=0)

                        pred_labels = [
                            np.full(bbox.shape[0], i, dtype=np.int32)
                            for i, bbox in enumerate(bbox_result)
                        ]
                        pred_labels = np.concatenate(pred_labels, axis=0)

                        if score_thr > 0:
                            assert bboxes.shape[1] == 5
                            scores = bboxes[:, -1]
                            inds = scores > score_thr
                            bboxes = bboxes[inds, :]
                            pred_labels = pred_labels[inds]

                        # 过滤那些框的宽高不合理的框
                        if hw_thr > 0 and len(bboxes) > 0:
                            ws = bboxes[:, 2] - bboxes[:, 0]
                            hs = bboxes[:, 3] - bboxes[:, 1]
                            inds = np.where((hs > hw_thr) & (ws > hw_thr))[0]
                            bboxes = bboxes[inds, :]
                            pred_labels = pred_labels[inds]

                        pred = torch.from_numpy(np.concatenate([bboxes, pred_labels.reshape(-1, 1)], axis=1))  # xyxy,score,cls

                        # pred is [xyxy, conf, pred_label]
                        pred_per_image.append(pred)

                    sub_preds += pred_per_image

                print('merge')

                # 合并子图上的检测，再次进行nms
                newpred = []
                for det, (x, y) in zip(sub_preds, dataset.start_positions):
                    if len(det):
                        det[:, [0, 2]] += x
                        det[:, [1, 3]] += y
                        newpred.append(det)

                # Apply Classifier
                if classify and len(newpred) > 0:
                    print('before classify: %d' % (len(newpred)))
                    newpred = apply_classifier(newpred, modelc, dataset.img0, dataset.img0, ti, oi,
                                               save_root=str(save_dir))
                    print('after classify: %d' % (len(newpred)))

                if len(newpred) > 0:
                    sub_preds = torch.cat(newpred)
                else:
                    sub_preds = []

                print('draw dets')
                im0 = dataset.img0
                tl = 3 or round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # line/font thickness

                if view_img and len(sub_preds) > 0:

                    for i, det in enumerate(sub_preds):  # xyxy, score, label
                        xmin, ymin, xmax, ymax, conf, label = det
                        label = int(label)
                        if label in shown_labels:
                            cv2.rectangle(im0, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                          color=colors[label],
                                          thickness=2, lineType=cv2.LINE_AA)
                            label_txt = f'{names[label]} {conf:.2f}'
                            cv2.putText(im0, label_txt, (int(xmin), int(ymin) - 2), 0, tl / 3,
                                        color=colors[label], thickness=2, lineType=cv2.LINE_AA)

                print('draw gt')
                if view_img and len(gt_boxes) > 0:
                    for box, label in zip(gt_boxes.cpu().numpy().copy(),
                                          gt_labels.cpu().numpy().copy()):  # xyxy, score, label
                        xmin, ymin, xmax, ymax = box

                        xmin -= xoffset
                        ymin -= yoffset
                        xmax -= xoffset
                        ymax -= yoffset

                        if xmin >= 0 and ymin >= 0:
                            print([xmin, ymin, xmax, ymax, label])
                            label = int(label) - 1
                            cv2.rectangle(im0, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                          color=(255,255,255),
                                          thickness=2, lineType=cv2.LINE_AA)
                            # label_txt = f'{names[label]} {conf:.2f}'
                            # cv2.putText(im0, label_txt, (int(xmin), int(ymin) - 2), 0, tl / 3,
                            #             color=[225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                if view_img:
                    print('write image data')
                    for b in range(3):
                        band = outdata.GetRasterBand(b + 1)
                        band.WriteArray(im0[:sub_height, :sub_width, b], xoff=xoffset, yoff=yoffset)
                        band.SetNoDataValue(0)
                        band.FlushCache()
                        del band
                # outdata.WriteRaster(xoffset, yoffset, sub_width, sub_height,
                #                im0[:sub_height, :sub_width].tobytes(),
                #                sub_width, sub_height, band_list=[1,2,3])
                # outdata.FlushCache()

                print('save tmp image data')
                # for si, (left, up) in enumerate(dataset.start_positions):
                #     right = min(sub_width, left + imgsz)
                #     bottom = min(sub_height, up + imgsz)
                #     cv2.imwrite(str(save_dir) + '/' + file_prefix + '_result_%d_%d.png' % (oi, si),
                #                 im0[up:bottom, left:right, ::-1])

                # ii = 0
                # for sx in range(0, sub_width, gt_subsize):
                #     for sy in range(0, sub_height, gt_subsize):
                #         subimg = im0[sy:(sy + gt_subsize), sx:(sx + gt_subsize), ::-1]
                #         minval = subimg.min()
                #         maxval = subimg.max()
                #         if maxval > minval:
                #             cv2.imwrite(str(save_dir) + '/' + file_prefix + '_result_%d_%d.png' % (oi, ii),
                #                         subimg)
                #             ii += 1

                if len(sub_preds) > 0:
                    sub_preds[:, [0, 2]] += xoffset
                    sub_preds[:, [1, 3]] += yoffset
                    all_preds.append(sub_preds)

                del dataset.img0
                del dataset
                import gc
                gc.collect()

            if view_img:
                outdata.FlushCache()
                del outdata
                del driver

                final_save_path = save_path.replace('_uncompressed.tif', '.tif')

                print('compressing the result file')
                time.sleep(3)
                command = 'gdal_translate -of GTiff -co "TILED=YES" -co "COMPRESS=LZW" -co "BIGTIFF=YES" %s %s' % \
                          (save_path, final_save_path)
                os.system(command)

                if os.path.exists(final_save_path):
                    time.sleep(2)
                    # os.system('rm -rf %s' % save_path)
                    os.remove(save_path)


            all_preds = torch.cat(all_preds)  # xmin, ymin, xmax, ymax, score, label

            torch.save(all_preds, all_preds_filename)

        # 过滤那些框的宽高不合理的框
        if hw_thr > 0 and len(all_preds) > 0:
            ws = all_preds[:, 2] - all_preds[:, 0]
            hs = all_preds[:, 3] - all_preds[:, 1]
            inds = np.where((hs > hw_thr) & (ws > hw_thr))[0]
            all_preds = all_preds[inds, :]

        all_preds_before = all_preds.clone()

        all_preds = all_preds.to(device)
        # 只保留0.5得分的框
        # all_preds = all_preds[all_preds[:, 4] >= 0.5, :]
        # TODO zzs 这里需要在全图范围内再来一次nms
        # 每个类进行nms
        tmp_preds = []
        all_preds_cpu = all_preds.cpu().numpy()
        for label in [0, 1, 2, 3]:
            idx = np.where(all_preds_cpu[:, 5] == label)[0]
            if len(idx) > 0:
                # if label == 0:
                #     pw = all_preds_cpu[idx, 2] - all_preds_cpu[idx, 0]
                #     ph = all_preds_cpu[idx, 3] - all_preds_cpu[idx, 1]
                #     inds = np.where((pw >= 100) | (ph >= 100))[0]
                #     valid_inds = idx[inds]
                # elif label == 1:
                #     pw = all_preds_cpu[idx, 2] - all_preds_cpu[idx, 0]
                #     ph = all_preds_cpu[idx, 3] - all_preds_cpu[idx, 1]
                #     inds = np.where((pw < 100) & (ph < 100))[0]
                #     valid_inds = idx[inds]
                valid_inds = idx
                dets = all_preds_cpu[valid_inds, :5]
                keep = py_cpu_nms(dets, thresh=0.5)
                tmp_preds.append(all_preds[valid_inds[keep]])
        all_preds = torch.cat(tmp_preds)

        if len(all_preds) > 0:
            # 对杆塔目标进行聚类合并，减少一个杆塔多个预测框
            all_preds_cpu = all_preds.cpu().numpy()

            idx = np.where(all_preds_cpu[:, 5] == 0)[0]   # 0:小杆塔，1:中杆塔，2:大杆塔
            all_preds_small = all_preds_cpu[idx, :]
            all_preds_small = all_preds_small[all_preds_small[:, 4] > 0.5]  # TODO 这里阈值需要设定
            if len(all_preds_small) == 0:
                all_preds_small = np.empty((0, 6), dtype=all_preds_cpu.dtype)

            idx = np.where(all_preds_cpu[:, 5] == 1)[0]   # 0:小杆塔，1:中杆塔，2:大杆塔
            all_preds_mid = all_preds_cpu[idx, :]
            all_preds_mid = all_preds_mid[all_preds_mid[:, 4] > 0.5]  # TODO 这里阈值需要设定
            if len(all_preds_mid) == 0:
                all_preds_mid = np.empty((0, 6), dtype=all_preds_cpu.dtype)

            idx = np.where(all_preds_cpu[:, 5] == 2)[0]   # 0:小杆塔，1:中杆塔，2:大杆塔
            all_preds_0 = all_preds_cpu[idx, :]
            idx = np.where(all_preds_cpu[:, 5] == 3)[0]   # 3:绝缘子
            all_preds_1 = all_preds_cpu[idx, :]
            all_preds_1 = all_preds_1[all_preds_1[:, 4] > 0.5]  # TODO 这里阈值需要设定
            if len(all_preds_0) > 0:
                print('before cluster: %d' % (len(all_preds_0)))
                # 只对杆塔进行聚类
                xmin, ymin, xmax, ymax = np.split(all_preds_0[:, :4], 4, axis=1)
                xc = (xmin + xmax) / 2
                yc = (ymin + ymax) / 2
                ws = xmax - xmin
                hs = ymax - ymin
                estimator = DBSCAN(eps=max(ws.mean(), hs.mean()) * 1.5, min_samples=1)
                X = np.concatenate([xc, yc], axis=1)  # N x 2

                estimator.fit(X)
                ##初始化一个全是False的bool类型的数组
                core_samples_mask = np.zeros_like(estimator.labels_, dtype=bool)
                '''
                   这里是关键点(针对这行代码：xy = X[class_member_mask & ~core_samples_mask])：
                   db.core_sample_indices_  表示的是某个点在寻找核心点集合的过程中暂时被标为噪声点的点(即周围点
                   小于min_samples)，并不是最终的噪声点。在对核心点进行联通的过程中，这部分点会被进行重新归类(即标签
                   并不会是表示噪声点的-1)，也可也这样理解，这些点不适合做核心点，但是会被包含在某个核心点的范围之内
                '''
                core_samples_mask[estimator.core_sample_indices_] = True

                ##每个数据的分类
                cluster_lables = estimator.labels_
                unique_labels = set(cluster_lables)
                ##分类个数：lables中包含-1，表示噪声点
                n_clusters_ = len(np.unique(cluster_lables)) - (1 if -1 in cluster_lables else 0)

                newpred = []
                for k in unique_labels:
                    # ##-1表示噪声点,这里的k表示黑色
                    # if k == -1:
                    #     col = 'k'

                    ##生成一个True、False数组，lables == k 的设置成True
                    class_member_mask = (cluster_lables == k)
                    indices = class_member_mask & core_samples_mask

                    ##两个数组做&运算，找出即是核心点又等于分类k的值  markeredgecolor='k',
                    # xy = X[class_member_mask & core_samples_mask]
                    # plt.plot(xy[:, 0], xy[:, 1], 'o', c=col, markersize=14)
                    '''
                       1)~优先级最高，按位对core_samples_mask 求反，求出的是噪音点的位置
                       2)& 于运算之后，求出虽然刚开始是噪音点的位置，但是重新归类却属于k的点
                       3)对核心分类之后进行的扩展
                    '''
                    # xy = X[class_member_mask & ~core_samples_mask]
                    # plt.plot(xy[:, 0], xy[:, 1], 'o', c=col, markersize=6)

                    tmppred = all_preds_0[indices]
                    xmin = np.min(tmppred[:, 0])
                    ymin = np.min(tmppred[:, 1])
                    xmax = np.max(tmppred[:, 2])
                    ymax = np.max(tmppred[:, 3])
                    score = np.max(tmppred[:, 4])
                    print(k, score)
                    newpred.append([xmin, ymin, xmax, ymax, score, 2])
                if len(newpred) > 0:
                    all_preds_0_clustered = np.array(newpred)
                    # 使用规则去掉不合理的杆塔 TODO 可能需要修改规则
                    ws = all_preds_0_clustered[:, 2] - all_preds_0_clustered[:, 0]
                    hs = all_preds_0_clustered[:, 3] - all_preds_0_clustered[:, 1]
                    wsmean, hsmean = ws.mean(), hs.mean()
                    thres = min(wsmean, hsmean)
                    # TODO 这里的规则就是检测出来的杆塔长宽要满足一个条件，否则去掉那个检测
                    inds = np.where((ws > 0.25 * thres) & (hs > 0.25 * thres))[0]
                    all_preds_0_clustered = all_preds_0_clustered[inds]
                    all_preds_0_clustered = all_preds_0_clustered[all_preds_0_clustered[:, 4] > 0.15]  # TODO 这里阈值需要设定

                    # # 这里再次进行分类
                    # if classify and len(all_preds_0_clustered) > 0:
                    #     all_preds_0_clustered1 = torch.from_numpy(all_preds_0_clustered)
                    #     ims = np.zeros((len(all_preds_0_clustered1), 224, 224, 3), dtype=np.uint8)
                    #     b = xyxy2xywh(all_preds_0_clustered1[:, :4])
                    #     b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
                    #     b[:, 2:] = b[:, 2:] * 1.3 + 32  # pad
                    #     for ii, pred in enumerate(b):
                    #         cutout = []
                    #         xc, yc, w1, h1 = pred
                    #         xc, yc = int(xc), int(yc)
                    #         w1, h1 = int(w1), int(h1)
                    #         xoffset = max(0, xc - w1 // 2)
                    #         yoffset = max(0, yc - h1 // 2)
                    #         if xoffset + w1 > orig_width:
                    #             w1 = orig_width - xoffset
                    #         if yoffset + h1 > orig_height:
                    #             h1 = orig_height - yoffset
                    #         for bi in range(3):
                    #             band = ds.GetRasterBand(bi + 1)
                    #             band_data = band.ReadAsArray(xoffset, yoffset, win_xsize=w1, win_ysize=h1)
                    #             cutout.append(band_data)
                    #         cutout = np.stack(cutout, -1)  # RGB
                    #         # cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                    #         ims[ii] = cv2.resize(cutout, (224, 224))  # BGR
                    #         cv2.imwrite('test%d_%d.jpg' % (ti, ii), ims[ii, :, :, ::-1])
                    #     ims = ims.transpose([0, 3, 1, 2])  # RGB, to 3x416x416
                    #     ims = np.ascontiguousarray(ims, dtype=np.float32)  # uint8 to float32
                    #     ims /= 255.0  # 0 - 255 to 0.0 - 1.0
                    #
                    #     dtype = torch.float32
                    #     mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=dtype, device=device)
                    #     std = torch.as_tensor([0.229, 0.224, 0.225], dtype=dtype, device=device)
                    #     if (std == 0).any():
                    #         raise ValueError(
                    #             'std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
                    #     if mean.ndim == 1:
                    #         mean = mean.view(-1, 1, 1)
                    #     if std.ndim == 1:
                    #         std = std.view(-1, 1, 1)
                    #
                    #     ims = torch.Tensor(ims).to(all_preds.device)
                    #     ims.sub_(mean).div_(std)
                    #     pred_0 = []
                    #     for im in ims:
                    #         pred_0.append(modelc(im[None]).argmax(1))  # classifier prediction
                    #     pred_0 = 1 - torch.cat(pred_0)  # 0 is pos, 1 is neg
                    #
                    #     valid_indices = []
                    #     for ii, pred_label in enumerate(pred_0):
                    #         if pred_label == 0:
                    #             valid_indices.append(ii)
                    #     print(all_preds_0_clustered)
                    #     print(valid_indices)
                    #     all_preds_0_clustered = all_preds_0_clustered[valid_indices]
                else:
                    all_preds_0_clustered = []

                print('after cluster: %d' % (len(all_preds_0_clustered)))
            else:
                all_preds_0_clustered = []

            if len(all_preds_0_clustered) > 0 and len(all_preds_1) > 0:
                all_preds_cpu = np.concatenate([all_preds_0_clustered, all_preds_1], axis=0)
            elif len(all_preds_0_clustered) > 0 and len(all_preds_1) == 0:
                all_preds_cpu = all_preds_0_clustered
            elif len(all_preds_0_clustered) == 0 and len(all_preds_1) > 0:
                all_preds_cpu = all_preds_1
            else:
                all_preds_cpu = np.empty((0, 6), dtype=all_preds_cpu.dtype)

            all_preds_cpu = np.concatenate([all_preds_cpu, all_preds_small, all_preds_mid], axis=0)
            all_preds = torch.from_numpy(all_preds_cpu).to(all_preds.device)

        nl = len(gt_labels)
        if nl:
            for box, label in zip(gt_boxes, gt_labels):
                xmin, ymin, xmax, ymax = box
                w1 = xmax - xmin
                h1 = ymax - ymin
                # for coco format
                single_obj = {'area': int(w1 * h1),
                              'category_id': int(label),  # starting from 1
                              'segmentation': []}
                single_obj['segmentation'].append(
                    [int(xmin), int(ymin), int(xmax), int(ymax),
                     int(xmax), int(ymax), int(xmin), int(ymax)]
                )
                single_obj['iscrowd'] = 0

                single_obj['bbox'] = int(xmin), int(ymin), int(w1), int(h1)
                single_obj['image_id'] = image_id
                single_obj['id'] = inst_count
                gt_json_dict['annotations'].append(single_obj)
                inst_count = inst_count + 1

            single_image = {}
            single_image['file_name'] = file_prefix + '.tif'
            single_image['id'] = image_id
            single_image['width'] = orig_width
            single_image['height'] = orig_height
            gt_json_dict['images'].append(single_image)

            # Append to pycocotools JSON dictionary
        if True:
            # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
            for p in all_preds.tolist():
                xmin, ymin, xmax, ymax, score, label = p
                w = xmax - xmin
                h = ymax - ymin
                b = [xmin, ymin, w, h]
                jdict.append({'image_id': image_id,
                              'category_id': int(label) + 1,  # starting from 1
                              'bbox': [round(x, 3) for x in b],
                              'score': round(score, 5)})
            # for box, label in zip(gt_boxes.tolist(),
            #                       gt_labels.tolist()):
            #     xmin, ymin, xmax, ymax = box
            #     w1 = xmax - xmin
            #     h1 = ymax - ymin
            #     b = [xmin, ymin, w1, h1]
            #     score = 0.9
            #     jdict.append({'image_id': image_id,
            #                   'category_id': int(label),
            #                   'bbox': [round(x, 3) for x in b],
            #                   'score': round(score, 5)})

        # Assign all predictions as incorrect
        seen += 1

        if nl: # if have gt_boxes
            correct = torch.zeros(all_preds.shape[0], niou, dtype=torch.bool, device=device)
            gt_labels = gt_labels - 1
            tcls = gt_labels.tolist() if nl else []  # target class
            detected = []  # target indices
            tcls_tensor = gt_labels.reshape(-1)

            # target boxes
            tbox = gt_boxes
            if is_plot:
                confusion_matrix.process_batch(all_preds,
                                               torch.cat((gt_labels.reshape([-1, 1]),
                                                          tbox), axis=1))

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                pi = (cls == all_preds[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(all_preds[pi, :4], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), all_preds[:, 4].cpu(), all_preds[:, 5].cpu(), tcls))

        save_predictions_to_envi_xml(preds=all_preds,
                                     save_xml_filename=str(save_dir) + '/' + file_prefix + '.xml',
                                     gdal_proj_info=projection_esri,
                                     gdal_trans_info=geotransform,
                                     names=names,
                                     colors={0: "255,0,0", 1: "0,0,255", 2: "0,255,255", 3: "255,255,0"})
        save_predictions_to_envi_xml(preds=all_preds_before,
                                     save_xml_filename=str(save_dir) + '/' + file_prefix + '_before.xml',
                                     gdal_proj_info=projection_esri,
                                     gdal_trans_info=geotransform,
                                     names=names,
                                     colors={0: "255,0,0", 1: "0,0,255", 2: "0,255,255", 3: "255,255,0"})


    pred_json = str(save_dir) + '/all_preds.json'
    with open(pred_json, 'w') as f:
        json.dump(jdict, f, indent=4)

    if nl:  # if have gt_boxes
        gt_json = str(save_dir) + '/all_gt.json'
        with open(gt_json, 'w') as f:
            json.dump(gt_json_dict, f, indent=4)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(gt_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
            print('results from coco: ', map, map50)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=is_plot, save_dir=save_dir, names=names)

            print('ap: ', ap)

            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            print(mp, mr, map50, map)
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        print(s)
        lines = []
        lines.append(s+'\n')
        pf = '%20s' + '%12.3g' * 6  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
        lines.append(pf % ('all', seen, nt.sum(), mp, mr, map50, map) + '\n')

        # Print results per class
        if nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
                lines.append(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]) + '\n')
        print('finally, get it done!')

        if len(lines):
            all_stats_filename = str(save_dir) + '/all_stats.txt'
            for line in lines:
                print(line.replace('\n',''))
            with open(all_stats_filename, 'w') as fp:
                fp.writelines(lines)
        # import pdb
        # pdb.set_trace()


if __name__ == '__main__':
    main()

from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import glob,os
import mmcv
import numpy as np
import torch
import cv2


def main():
    parser = ArgumentParser()
    parser.add_argument('source', help='image path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--save_root', help='where to save')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    class_names = ['gan', 'jyz']

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    files = glob.glob(args.source + '/*.tif')

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)
    num_files = len(files)
    for index, file in enumerate(files):
        prefix = file.split(os.sep)[-1].replace('.tif', '')
        save_filename = os.path.join(args.save_root, prefix+'.tif')

        # test a single image
        result = inference_detector(model, file)
        # show the results
        # show_result_pyplot(model, args.img, result, score_thr=args.score_thr)

        img = mmcv.imread(file)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)

        score_thr = args.score_thr
        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            if segms is not None:
                segms = segms[inds, ...]

        img = mmcv.bgr2rgb(img)
        img = np.ascontiguousarray(img)

        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            bbox_int = bbox.astype(np.int32)
            label_text = class_names[
                label] if class_names is not None else f'class {label}'
            if len(bbox) > 4:
                label_text += f'|{bbox[-1]:.02f}'
            cv2.rectangle(img,
                          (bbox_int[0], bbox_int[1]),
                          (bbox_int[2], bbox_int[3]), color=(0, 0, 255) if label==0 else (0, 255, 0),
                          thickness=2)
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1]), fontFace=1, fontScale=2, color=(0,255,255))
        img = mmcv.rgb2bgr(img)
        cv2.imwrite(save_filename, img)

        print("%d/%d %s Done" % (index, num_files, file))


if __name__ == '__main__':
    main()

import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, box_utils
from pcdet.datasets.kitti.kitti_object_eval_python.eval import print_str


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
            #############################################################
            # print(f'pred_dicts length: %d' % len(pred_dicts))
            # for key, value in pred_dicts[0].items():
            #     print(key, type(value), value.shape)
            # print(f'ret_dict length: %d' % len(ret_dict))
            # for key, value in ret_dict[0].items():
            #     print(key, type(value), value.shape)
            #############################################################
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def eval_one_epoch_seg(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dict = model(batch_dict)
            # for key, value in pred_dict.items():
            #     if key != 'batch_size':
            #         print(key, type(value), value.shape)
            #     else:
            #         print(key, value)
        annos = []
        pts_num = int(len(pred_dict['points'])/pred_dict['batch_size'])
        points = pred_dict['point_coords'].cpu().numpy()
        gt_boxes = pred_dict['gt_boxes'].cpu().numpy()
        point_cls_scores = pred_dict['point_cls_scores'].cpu().numpy()
        for j in range(pred_dict['batch_size']):
            boxes = gt_boxes[j]
            boxes = boxes[~np.all(boxes == 0, axis=1)]
            annos.append({
                'point_coords': points[j*pts_num:(j+1)*pts_num, :],
                'gt_boxes': boxes,
                'point_cls_scores': point_cls_scores[j*pts_num:(j+1)*pts_num]
            })
        det_annos += annos

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = point_seg_evaluation(dataset, det_annos, class_names, output_path=final_output_dir)

    logger.info(result_str)
    #ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return result_dict


def point_seg_evaluation(dataset, det_dicts, classnames, output_path):
    result_str = ''
    result_dict = {}
    total_correct = 0
    total_seen = 0
    total_correct_class = [0 for _ in classnames]
    total_seen_class = [0 for _ in classnames]
    total_iou_class = [0 for _ in classnames]
    for det in det_dicts:
        #point_cls_labels = generate_point_label(det['frame_id'], det['point_coords'], det['gt_boxes'])
        point_cls_labels = np.zeros((len(det['point_coords'])))
        for i,box in enumerate(det['gt_boxes']):
            box_dim = box[np.newaxis, :]
            box_dim = box_dim[:, 0:7]
            corners = box_utils.boxes_to_corners_3d(box_dim)
            corners = np.squeeze(corners, axis=0)
            flag = box_utils.in_hull(det['point_coords'][:, 1:], corners)
            point_cls_labels[flag] = int(box[-1])
        total_correct += np.sum(det['point_cls_scores'] == point_cls_labels)
        total_seen += det['point_cls_scores'].size
        for i in range(len(classnames)):
            total_seen_class[i] += np.sum((point_cls_labels == i+1))
            total_correct_class[i] += np.sum((det['point_cls_scores'] == i+1) & (point_cls_labels == i+1))
            total_iou_class += np.sum((det['point_cls_scores'] == i+1) | (point_cls_labels == i+1))
    mIoU = np.mean(np.array(total_correct_class)/(np.array(total_iou_class, dtype=np.float) + 1e-6))
    total_correct /= total_seen
    #total_correct_class /= np.mean(np.array(total_correct_class)/(np.array(total_seen_class, dtype=np.float) + 1e-6))
    result_str += print_str((f"point avg class IoU: {mIoU:.4f}"))
    result_str += print_str((f"point accuracy: {total_correct:.4f}"))
    #result_str += print_str((f"point avg class acc: {total_correct_class:.4f}"))
    result_str += print_str(f"car acc: {total_correct_class[0]/total_seen_class[0]:.4f}")
    result_str += print_str(f"Pedestrian acc: {total_correct_class[1] / total_seen_class[1]:.4f}")
    result_str += print_str(f"Cyclist acc: {total_correct_class[2] / total_seen_class[2]:.4f}")

    result_dict['mIoU'] = mIoU
    result_dict['avg_acc'] = total_correct
    result_dict['avg_car_acc'] = total_correct_class[0]/total_seen_class[0]
    result_dict['avg_ped_acc'] = total_correct_class[1] / total_seen_class[1]
    result_dict['avg_cyc_acc'] = total_correct_class[2] / total_seen_class[2]
    return result_str, result_dict


if __name__ == '__main__':
    pass

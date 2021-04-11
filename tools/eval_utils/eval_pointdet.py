import pickle
import time
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models.model_utils import model_nms_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils




def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

def merge_pred_results(pred_list):
    frame_id_list = [p['frame_id'] for p in pred_list]
    frame_id_set = set(frame_id_list)

    # batch_cls_preds: (B, num_boxes, num_classes)
    # batch_box_preds: (B, num_boxes, 7 + C)
    batch_cls_preds = []
    batch_box_preds = []
    batch_index = []
    frame_id = [None] * len(frame_id_set)
    for i, frame in enumerate(frame_id_set):
        box_preds = [p['box_pred'] for p in pred_list if p['frame_id'] == frame]
        cls_preds = [p['cls_pred'] for p in pred_list if p['frame_id'] == frame]
        box_preds = torch.cat(box_preds)
        cls_preds = torch.cat(cls_preds)
        index = [i] * len(box_preds)
        frame_id[i] = frame

        batch_box_preds.append(box_preds)
        batch_cls_preds.append(cls_preds)
        batch_index.append(index)

    pred_dict = {
        'batch_box_preds': torch.cat(batch_box_preds),
        'batch_cls_preds': torch.cat(batch_cls_preds),
        'batch_index': torch.cat(batch_index),
        'frame_id': frame_id
    }
    return pred_dict

def post_processing(model_cfg, batch_dict, num_class=3):
    """
    Args:
        batch_dict:
            batch_size:
            batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                            or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
            multihead_label_mapping: [(num_class1), (num_class2), ...]
            batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
            cls_preds_normalized: indicate whether batch_cls_preds is normalized
            batch_index: optional (N1+N2+...)
            has_class_labels: True/False
            roi_labels: (B, num_rois)  1 .. num_classes
            batch_pred_labels: (B, num_boxes, 1)
    Returns:

    """
    print('**********************batch_dict info*********************')
    for key, value in batch_dict.items():
        print(key)
        if key == 'batch_size' or key == 'frame_id':
            print(value)
        elif key == 'batch_box_preds' or key == 'batch_cls_preds':
            print(value.shape)
        # if key != 'points':
        #     print(value)
    print('**********************************************************')
    post_process_cfg = model_cfg.POST_PROCESSING
    batch_size = batch_dict['batch_size']
    recall_dict = {}
    pred_dicts = []
    for index in range(batch_size):
        if batch_dict.get('batch_index', None) is not None:
            assert batch_dict['batch_box_preds'].shape.__len__() == 2
            batch_mask = (batch_dict['batch_index'] == index)
        else:
            assert batch_dict['batch_box_preds'].shape.__len__() == 3
            batch_mask = index

        box_preds = batch_dict['batch_box_preds'][batch_mask]
        src_box_preds = box_preds

        if not isinstance(batch_dict['batch_cls_preds'], list):
            cls_preds = batch_dict['batch_cls_preds'][batch_mask]

            src_cls_preds = cls_preds
            assert cls_preds.shape[1] in [1, num_class]

            if not batch_dict['cls_preds_normalized']:
                cls_preds = torch.sigmoid(cls_preds)
        else:
            cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
            src_cls_preds = cls_preds
            if not batch_dict['cls_preds_normalized']:
                cls_preds = [torch.sigmoid(x) for x in cls_preds]

        if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
            if not isinstance(cls_preds, list):
                cls_preds = [cls_preds]
                multihead_label_mapping = [torch.arange(1, num_class, device=cls_preds[0].device)]
            else:
                multihead_label_mapping = batch_dict['multihead_label_mapping']

            cur_start_idx = 0
            pred_scores, pred_labels, pred_boxes = [], [], []
            for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                    cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )
                cur_pred_labels = cur_label_mapping[cur_pred_labels]
                pred_scores.append(cur_pred_scores)
                pred_labels.append(cur_pred_labels)
                pred_boxes.append(cur_pred_boxes)
                cur_start_idx += cur_cls_preds.shape[0]

            final_scores = torch.cat(pred_scores, dim=0)
            final_labels = torch.cat(pred_labels, dim=0)
            final_boxes = torch.cat(pred_boxes, dim=0)
        else:
            cls_preds, label_preds = torch.max(cls_preds, dim=-1)
            if batch_dict.get('has_class_labels', False):
                label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                label_preds = batch_dict[label_key][index]
            else:
                label_preds = label_preds + 1
            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                box_scores=cls_preds, box_preds=box_preds,
                nms_config=post_process_cfg.NMS_CONFIG,
                score_thresh=post_process_cfg.SCORE_THRESH
            )

            if post_process_cfg.OUTPUT_RAW_SCORE:
                max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                selected_scores = max_cls_preds[selected]

            final_scores = selected_scores
            final_labels = label_preds[selected]
            final_boxes = box_preds[selected]

        recall_dict = generate_recall_record(
            box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
            recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
            thresh_list=post_process_cfg.RECALL_THRESH_LIST
        )

        record_dict = {
            'pred_boxes': final_boxes,
            'pred_scores': final_scores,
            'pred_labels': final_labels
        }
        pred_dicts.append(record_dict)

    return pred_dicts, recall_dict

def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
    if 'gt_boxes' not in data_dict:
        return recall_dict

    rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
    gt_boxes = data_dict['gt_boxes'][batch_index]

    if recall_dict.__len__() == 0:
        recall_dict = {'gt': 0}
        for cur_thresh in thresh_list:
            recall_dict['roi_%s' % (str(cur_thresh))] = 0
            recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

    cur_gt = gt_boxes
    k = cur_gt.__len__() - 1
    while k > 0 and cur_gt[k].sum() == 0:
        k -= 1
    cur_gt = cur_gt[:k + 1]

    if cur_gt.shape[0] > 0:
        if box_preds.shape[0] > 0:
            iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
        else:
            iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

        if rois is not None:
            iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

        for cur_thresh in thresh_list:
            if iou3d_rcnn.shape[0] == 0:
                recall_dict['rcnn_%s' % str(cur_thresh)] += 0
            else:
                rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
            if rois is not None:
                roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

        recall_dict['gt'] += cur_gt.shape[0]
    else:
        gt_iou = box_preds.new_zeros(box_preds.shape[0])
    return recall_dict

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
    preds_list = []
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dict, _, _ = model(batch_dict)
            if pred_dict:
                preds_list.append(pred_dict)

    ############################################################
    print(f'pred_dicts length: %d' % len(preds_list))
    for key, value in preds_list[0].items():
        print(key, value)
    ############################################################
    disp_dict = {}
    preds = merge_pred_results(preds_list)  # merge all detection results from the same pc frame_id together
    pred_dicts, ret_dict = post_processing(cfg.MODEL, preds)  # nms
    statistics_info(cfg, ret_dict, metric, disp_dict)
    det_annos = dataset.generate_prediction_dicts(
        batch_dict, pred_dicts, class_names,
        output_path=final_output_dir if save_to_file else None
    )


        # disp_dict = {}
        #
        # statistics_info(cfg, ret_dict, metric, disp_dict)
        # annos = dataset.generate_prediction_dicts(
        #     batch_dict, pred_dicts, class_names,
        #     output_path=final_output_dir if save_to_file else None
        # )
        # det_annos += annos
        # if cfg.LOCAL_RANK == 0:
        #     progress_bar.set_postfix(disp_dict)
        #     progress_bar.update()


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



if __name__ == '__main__':
    pass

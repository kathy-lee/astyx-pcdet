import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pcdet.utils.box_utils import in_hull, boxes_to_corners_3d
from pcdet.models.model_utils.model_nms_utils import class_agnostic_nms
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8  # one cluster for each type


def make_fc_layers(fc_list, input_channels, output_channels):
    fc_layers = []
    c_in = input_channels
    for k in range(0, fc_list.__len__()):
        fc_layers.extend([
            nn.Linear(c_in, fc_list[k], bias=False),
            nn.BatchNorm1d(fc_list[k]),
            nn.ReLU(),
        ])
        c_in = fc_list[k]
    fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
    return nn.Sequential(*fc_layers)


class PointNetv1(nn.Module):
    def __init__(self, model_cfg, num_classes, input_channels=4):
        super(PointNetv1, self).__init__()
        self.model_cfg = model_cfg
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.cls_layers = make_fc_layers(fc_list=model_cfg, input_channels=1024, output_channels=num_classes)
        self.n_sample = 256
        self.pos_iou_thresh = 0.7
        self.neg_iou_thresh = 0.3
        self.pos_ratio = 0.5

    def forward(self, batch_data):  # bs,4,n
        print(batch_data.shape)
        out1 = F.relu(self.bn1(self.conv1(batch_data)))  # bs,64,n
        out2 = F.relu(self.bn2(self.conv2(out1)))  # bs,64,n
        out3 = F.relu(self.bn3(self.conv3(out2)))  # bs,64,n
        out4 = F.relu(self.bn4(self.conv4(out3)))  # bs,128,n
        out5 = F.relu(self.bn5(self.conv5(out4)))  # bs,1024,n
        global_feat = torch.max(out5, 2, keepdim=True)[0]  # bs,1024,1
        print(global_feat.size())
        logits = self.cls_layers(np.squeeze(global_feat, axis=2))
        softmax = nn.Softmax(dim=1)
        cls_score = softmax(logits)
        _, cls_pred = torch.max(cls_score, 1)
        one_hot_vec = torch.zeros(batch_data.shape[0], self.num_classes)
        one_hot_vec[:, cls_pred] = 1

        feature_dict = {
            'local_feat': out2,
            'global_feat': global_feat,
            'cls_pred': one_hot_vec[:, 0:3],
            'cls_logits': logits
        }
        return feature_dict

    def get_loss(self, pred, target):
        loss = F.cross_entropy(pred, target)
        return loss


class PointSeg(nn.Module):
    def __init__(self, model_cfg, num_classes, input_channels=4):
        super(PointSeg, self).__init__()
        self.model_cfg = model_cfg
        self.num_classes = num_classes

        self.dconv1 = nn.Conv1d(1088 + num_classes, 512, 1)
        self.dconv2 = nn.Conv1d(512, 256, 1)
        self.dconv3 = nn.Conv1d(256, 128, 1)
        self.dconv4 = nn.Conv1d(128, 128, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.dconv5 = nn.Conv1d(128, 2, 1)
        self.dbn1 = nn.BatchNorm1d(512)
        self.dbn2 = nn.BatchNorm1d(256)
        self.dbn3 = nn.BatchNorm1d(128)
        self.dbn4 = nn.BatchNorm1d(128)

    def forward(self, data_dict):  # bs,4,n
        bs = data_dict['batch_size']
        n_pts = data_dict['points'].size()[2]
        # print(data_dict['cls_pred'].size(), data_dict['cls_pred'].is_cuda)
        expand_one_hot_vec = data_dict['cls_pred'].view(bs, -1, 1).cuda()  # bs,3,1
        # print(data_dict['global_feat'].size(), data_dict['global_feat'].is_cuda)
        # print(expand_one_hot_vec.size(), expand_one_hot_vec.is_cuda)
        expand_global_feat = torch.cat([data_dict['global_feat'], expand_one_hot_vec], 1)  # bs,1027,1
        expand_global_feat_repeat = expand_global_feat.view(bs, -1, 1).repeat(1, 1, n_pts)  # bs,1027,n
        concat_feat = torch.cat([data_dict['local_feat'], expand_global_feat_repeat], 1)
        # bs, (641024+3)=1091, n

        x = F.relu(self.dbn1(self.dconv1(concat_feat)))  # bs,512,n
        x = F.relu(self.dbn2(self.dconv2(x)))  # bs,256,n
        x = F.relu(self.dbn3(self.dconv3(x)))  # bs,128,n
        x = F.relu(self.dbn4(self.dconv4(x)))  # bs,128,n
        x = self.dropout(x)
        x = self.dconv5(x)  # bs, 2, n

        seg_pred = x.transpose(2, 1).contiguous()  # bs, n, 2
        print('point segment prediction: ')
        print(seg_pred.size())
        data_dict.update({'seg_logits': seg_pred})
        data_dict.pop('global_feat')
        data_dict.pop('local_feat')
        return data_dict

    def get_loss(self, pred, target):
        logits = F.log_softmax(pred.view(-1, 2), dim=1)  # torch.Size([32768, 2])
        mask_label = target.view(-1).long()  # torch.Size([32768])
        loss = F.nll_loss(logits, mask_label)  # tensor(0.6361, grad_fn=<NllLossBackward>)
        return loss


class CenterRegNet(nn.Module):
    def __init__(self, n_classes=3):
        super(CenterRegNet, self).__init__()

        self.n_classes = n_classes

        self.conv1 = torch.nn.Conv1d(4, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        # self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(256 + self.n_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.fcbn1 = nn.BatchNorm1d(256)
        self.fcbn2 = nn.BatchNorm1d(128)

    def forward(self, data_dict):
        bs = data_dict['batch_size']
        print(data_dict['points'].size())
        x = F.relu(self.bn1(self.conv1(data_dict['points'])))  # bs,128,n
        x = F.relu(self.bn2(self.conv2(x)))  # bs,128,n
        x = F.relu(self.bn3(self.conv3(x)))  # bs,256,n
        x = torch.max(x, 2)[0]  # bs,256
        expand_one_hot_vec = data_dict['cls_pred'].view(bs, -1).cuda()  # bs,3
        x = torch.cat([x, expand_one_hot_vec], 1)  # bs,259
        x = F.relu(self.fcbn1(self.fc1(x)))  # bs,256
        x = F.relu(self.fcbn2(self.fc2(x)))  # bs,128
        x = self.fc3(x)  # bs,
        print(x.size())

        pts_new = data_dict['points']
        pts_new[:, :3, :] -= x.view(x.shape[0], -1, 1).repeat(1, 1, data_dict['points'].shape[-1])

        center_new = data_dict['center'] + x
        data_dict.update({'center': center_new})  # (32,3)
        data_dict.update({'points': pts_new})
        print(f'after center reg: %d' % bs)
        return data_dict

    def get_loss(self, center_pred, center_label):
        center_dist = torch.norm(center_pred - center_label, dim=1)  # (32,)
        center_loss = self.huber_loss(center_dist, delta=1.0)
        return center_loss


class BoxRegNet(nn.Module):
    def __init__(self, n_classes=3):
        """Amodal 3D Box Estimation Pointnet
        :param n_classes:3
        """
        super(BoxRegNet, self).__init__()
        self.n_classes = n_classes

        self.conv1 = nn.Conv1d(4, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512 + n_classes, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)

    def forward(self, data_dict):  # bs,3,m
        """
        :param
            data_dict['pts']: [bs,3,m]: x,y,z after InstanceSeg
        :return:
            box_pred: [bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4]
            including box centers, heading bin class scores and residuals,
            and size cluster scores and residuals
        """
        bs = data_dict['batch_size']

        out1 = F.relu(self.bn1(self.conv1(data_dict['points'])))  # bs,128,n
        out2 = F.relu(self.bn2(self.conv2(out1)))  # bs,128,n
        out3 = F.relu(self.bn3(self.conv3(out2)))  # bs,256,n
        out4 = F.relu(self.bn4(self.conv4(out3)))  # bs,512,n
        global_feat = torch.max(out4, 2, keepdim=False)[0]  # bs,512

        expand_one_hot_vec = data_dict['cls_pred'].view(bs, -1).cuda()  # bs,3
        print(global_feat.size(), expand_one_hot_vec.size())  # torch.Size([2, 512]) torch.Size([2, 2])
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec], 1)  # bs,515
        print(expand_global_feat.size())
        x = self.fc1(expand_global_feat)
        x = F.relu(self.fcbn1(self.fc1(expand_global_feat)))  # bs,512
        x = F.relu(self.fcbn2(self.fc2(x)))  # bs,256
        box_pred = self.fc3(x)  # bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4

        center_delta = box_pred[:, :3]
        center = data_dict['center']
        data_dict.update({'center_last': center})
        data_dict.update({'center': center + center_delta})
        data_dict.update({'box_reg_pred': box_pred})
        return data_dict

    def get_loss(self, center, batch_target, box_pred, corner_loss_weight=10.0):
        bs = box_pred.shape[0]

        center_label = batch_target['center_label']
        head_cls_label = batch_target['hclass']
        head_res_label = batch_target['hres']
        size_cls_label = batch_target['sclass']
        size_res_label = batch_target['sres']

        _, head_scores, head_res_norm, head_res, size_scores, size_res_norm, size_res = self.parse_to_tensors(box_pred)

        # center residual reg loss
        center_dist = torch.norm(center - center_label, dim=1)  # (32,)
        center_loss = self.huber_loss(center_dist, delta=2.0)

        # Heading Loss
        head_cls_loss = F.nll_loss(F.log_softmax(head_scores, dim=1), head_cls_label.long())  # tensor(2.4505, grad_fn=<NllLossBackward>)
        hcls_onehot = torch.eye(NUM_HEADING_BIN)[head_cls_label.long()].cuda()  # 32,12
        head_res_norm_label = head_res_label / (np.pi / NUM_HEADING_BIN)  # 32
        head_res_norm_dist = torch.sum(head_res_norm * hcls_onehot.float(), dim=1)  # 32
        ### Only compute reg loss on gt label
        head_res_norm_loss = self.huber_loss(head_res_norm_dist - head_res_norm_label, delta=1.0)

        # Size loss
        size_cls_loss = F.nll_loss(F.log_softmax(size_scores, dim=1), size_cls_label.long())  # tensor(2.0240, grad_fn=<NllLossBackward>)
        scls_onehot = torch.eye(NUM_SIZE_CLUSTER)[size_cls_label.long()].cuda()  # 32,8
        scls_onehot_repeat = scls_onehot.view(-1, NUM_SIZE_CLUSTER, 1).repeat(1, 1, 3)  # 32,8,3
        predicted_size_residual_normalized_dist = torch.sum(size_res_norm * scls_onehot_repeat.cuda(), dim=1)  # 32,3
        mean_size_arr_expand = torch.from_numpy(self.g_mean_size_arr).float().cuda().view(1, NUM_SIZE_CLUSTER, 3)#1,8,3
        mean_size_label = torch.sum(scls_onehot_repeat * mean_size_arr_expand, dim=1)  # 32,3
        size_res_label_norm = size_res_label / mean_size_label.cuda()
        size_normalized_dist = torch.norm(size_res_label_norm - predicted_size_residual_normalized_dist, dim=1)  # 32
        size_res_norm_loss = self.huber_loss(size_normalized_dist, delta=1.0)  # tensor(11.2784, grad_fn=<MeanBackward0>)

        # Corner Loss
        corners_3d = self.get_box3d_corners(center, head_res, size_res, self.g_mean_size_arr).cuda()  # (bs,NH,NS,8,3)(32, 12, 8, 8, 3)
        gt_mask = hcls_onehot.view(bs, NUM_HEADING_BIN, 1).repeat(1, 1, NUM_SIZE_CLUSTER) * \
                  scls_onehot.view(bs, 1, NUM_SIZE_CLUSTER).repeat(1, NUM_HEADING_BIN, 1)  # (bs,NH=12,NS=8)
        corners_3d_pred = torch.sum(gt_mask.view(bs, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1, 1) \
                                    .float().cuda() * corners_3d, dim=[1, 2])  # (bs,8,3)
        heading_bin_centers = torch.from_numpy(np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN)).float().cuda()  # (NH,)
        heading_label = head_res_label.view(bs, 1) + heading_bin_centers.view(1,NUM_HEADING_BIN)  # (bs,1)+(1,NH)=(bs,NH)
        heading_label = torch.sum(hcls_onehot.float() * heading_label, 1)
        mean_sizes = torch.from_numpy(self.g_mean_size_arr).float().view(1, NUM_SIZE_CLUSTER, 3).cuda()  # (1,NS,3)
        size_label = mean_sizes + size_res_label.view(bs, 1, 3)  # (1,NS,3)+(bs,1,3)=(bs,NS,3)
        size_label = torch.sum(scls_onehot.view(bs, NUM_SIZE_CLUSTER, 1).float() * size_label, axis=[1])  # (B,3)
        corners_3d_gt = self.get_box3d_corners_helper(center_label, heading_label, size_label)  # (B,8,3)
        corners_3d_gt_flip = self.get_box3d_corners_helper(center_label, heading_label + np.pi, size_label)  # (B,8,3)
        corners_dist = torch.min(torch.norm(corners_3d_pred - corners_3d_gt, dim=-1),
                                 torch.norm(corners_3d_pred - corners_3d_gt_flip, dim=-1))
        corners_loss = self.huber_loss(corners_dist, delta=1.0)

        loss = center_loss + head_cls_loss + size_cls_loss + (head_res_norm_loss + size_res_norm_loss) * 20 \
               + corner_loss_weight * corners_loss
        return loss


class PointNetDetector(nn.Module):
    def __init__(self, model_cfg, dataset, n_classes=3, n_channels=4):
        super(PointNetDetector, self).__init__()
        self.model_cfg = model_cfg
        self.n_classes = n_classes
        self.dataset = dataset
        self.PointCls = PointNetv1([256, 256], n_classes+1, n_channels)
        self.PointSeg = PointSeg([256, 256], n_classes, n_channels)
        self.CenterReg = CenterRegNet(n_classes=n_classes)
        self.BoxReg = BoxRegNet(n_classes=n_classes)
        self.NUM_OBJECT_POINT = 512
        self.NUM_SIZE_CLUSTER = 3

        # self.g_type2class = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        # g_class2type = {self.g_type2class[t]: t for t in self.g_type2class}
        # self.g_type_mean_size = {'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
        #                          'Pedestrian': np.array([0.84422524, 0.66068622, 1.76255119]),
        #                          'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127])}
        self.g_type_mean_size = np.array([[3.88311640418, 1.62856739989, 1.52563191462],
                                          [0.84422524, 0.66068622, 1.76255119],
                                          [1.76282397, 0.59706367, 1.73698127]])
        # self.g_mean_size_arr = np.zeros((self.NUM_SIZE_CLUSTER, 3))  # size clusters
        # for i in range(self.NUM_SIZE_CLUSTER):
        #     self.g_mean_size_arr[i, :] = self.g_type_mean_size[i]

    def forward(self, batch_dict):  # bs,4,n

        # proposals = self.generate_proposals(batch_dict)
        #
        # # 3D Proposal Classification PointNet
        # feature_dict = self.PointCls(proposals['pts'])
        #
        # # if self.training:
        # #     rois = self.get_rois(proposals, feature_dict, n_pre_nms=12000, n_post_nms=2000)
        # #     rois, gt_roi_loc, gt_roi_label = self.proposal_target_creator(rois, batch_dict)
        # # else:
        # #     rois = self.get_rois(proposals, feature_dict, n_pre_nms=6000, n_post_nms=300)
        # rois = self.get_rois(proposals, feature_dict, n_pre_nms=12000, n_post_nms=2000)
        #
        # # 3D Instance Segmentation PointNet
        # seg_logits = self.PointSeg(rois)  # bs,n,2
        #
        # # Mask Point Centroid
        # pts_xyz, mask_xyz_mean = self.point_cloud_masking(rois['pts'], seg_logits)  ###logits.detach()
        #
        # # Object Center Regression T-Net
        # pts_xyz = pts_xyz.cuda()
        # center_delta = self.CenterReg(pts_xyz)  # (32,3)
        # stage1_center = center_delta + mask_xyz_mean  # (32,3)
        # # if (np.isnan(stage1_center.cpu().detach().numpy()).any()):
        # #     ipdb.set_trace()
        # pts_xyz_new = pts_xyz - center_delta.view(center_delta.shape[0], -1, 1).repeat(1, 1, pts_xyz.shape[-1])
        #
        # # 3D Box Estimation
        # box_pred = self.BoxReg(pts_xyz_new)  # (32, 59)
        # center = box_pred[:, :3] + stage1_center  # bs,3

        # # rewrite forward
        # proposals = self.generate_proposals(batch_dict, self.model_cfg)  # proposals{'pts', 'frame_id', 'pos'}
        # feature_dict = self.PointCls(proposals['pts'])
        # rois = self.get_rois(proposals, feature_dict)  # rois{'pts', 'frame_id', 'pos', 'feature_loc', 'feature_glob', 'cls_pred', 'cls_logits'}
        # rois = self.PointSeg(rois)  # add rois{'seg_logits'}, deleted rois{'feature_loc', 'feature_glob'}
        # rois = self.point_cloud_masking(rois)  # add rois{'center'}
        # rois = self.CenterReg(rois)  # update rois{'center'} and {'pts'}
        # rois = self.BoxReg(rois)  # update rois{'center'}, add rois{ 'box_reg_pred' }, move old center in rois{'center_last'}

        # # rewrite forward
        # proposals = self.generate_proposals(batch_dict)
        # feature, cls_pred, cls_logits = self.PointCls(proposals)
        # rois = self.get_rois(proposals, feature, cls_pred) # rois{'pts', 'pos', 'frame_id'}
        # seg_logits = self.PointSeg(rois['pts'], cls_pred, feature)
        # rois, center_mask = self.point_cloud_masking(rois, seg_logits)
        # center_delta = self.CenterReg(rois['pts'], cls_pred)
        # rois['pts'] = rois['pts'] - center_delta.view(center_delta.shape[0], -1, 1).repeat(1, 1, rois['pts'].shape[-1])
        # box_reg_pred = self.BoxReg(rois['pts'], cls_pred)
        # center_pred = center_mask + center_delta + box_reg_pred[:, :3]

        # rewrite forward using proposal as batch data
        feature_dict = self.PointCls(batch_dict['points'])
        rois = self.get_rois(batch_dict, feature_dict)  # rois{'pts', 'frame_id', 'pos', 'feature_loc', 'feature_glob', 'cls_pred', 'cls_logits'}
        rois = self.PointSeg(rois)  # add rois{'seg_logits'}, deleted rois{'feature_loc', 'feature_glob'}
        rois = self.point_cloud_masking(rois)  # add rois{'center'}
        rois = self.CenterReg(rois)  # update rois{'center'} and {'pts'}
        rois = self.BoxReg(rois)
        print('Network forwarding finished.')
        if self.training:
            # batch_target = self.assign_targets(batch_dict, proposals, rois)
            batch_target = self.assign_targets2(batch_dict, rois)
            seg_loss_weight = 1.0
            box_loss_weight = 1.0
            cls_loss = self.PointCls.get_loss(feature_dict['cls_logits'], batch_target['cls_label'])
            seg_loss = self.PointSeg.get_loss(rois['seg_logits'], batch_target['point_label'])
            # center_reg_loss = self.CenterReg.get_loss(batch_target['center_label'], stage1_center)
            # box_reg_loss = self.BoxReg.get_loss(center, batch_target, box_pred)
            center_reg_loss = self.CenterReg.get_loss(rois['center_last'], batch_target['center_label'])
            box_reg_loss = self.BoxReg.get_loss(rois['center'], batch_target, rois['box_pred'])

            loss = cls_loss + seg_loss_weight * seg_loss + box_loss_weight * (center_reg_loss + box_reg_loss)
            tb_dict = {}
            disp_dict = {}
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def assign_targets2(self, batch_dict, rois):
        cls_label = self.assign_proposal_target2(batch_dict)
        point_label = self.assign_seg_target(rois)
        print('assign target:')
        print(cls_label.shape, point_label.shape)

        center_label = torch.zeros((rois['batch_size'], 3))
        size_cls_label = torch.zeros((rois['batch_size']), dtype=torch.int32)
        size_residual = torch.zeros((rois['batch_size'], 3))
        heading_cls_label = torch.zeros((rois['batch_size']), dtype=torch.int32)
        heading_residual = torch.zeros((rois['batch_size'], 3))
        for i in range(rois['batch_size']):
            # rois[i] with best matched gt box [k]
            k = 0
            center_label[i, :] = rois['gt_boxes'][i, k, :3]
            box_size = rois['gt_boxes'][i, k, 3:6]
            heading = rois['gt_boxes'][i, k, 6]
            size_cls_label[i] = rois['gt_boxes'][i, k, -1]
            size_residual[i, :] = box_size - self.g_type_mean_size[size_cls_label[i], :]
            angle = heading % (2 * np.pi)
            assert (angle >= 0 and angle <= 2 * np.pi)
            angle_per_class = 2 * np.pi / float(NUM_HEADING_BIN)
            shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
            heading_cls_label[i] = int(shifted_angle / angle_per_class)
            heading_residual[i, :] = shifted_angle - (heading_cls_label * angle_per_class + angle_per_class / 2)

        # center_label = rois['pos'][:, :3]
        # box_size = rois['pos'][:, 3:6]
        # heading = rois['pos'][6]
        #
        # center_label = batch_dict['gt_boxes'][:, :, :3]
        # box_size = batch_dict['gt_boxes'][:, :, 3:6]
        # heading = batch_dict['gt_boxes'][:, :, 6]  # bs,n_boxes

        # size_class = np.argmax(batch_dict['cls_pred'], axis=1)  # 0/1/2
        # size_residual = box_size - self.g_type_mean_size[size_class]

        # angle = heading % (2 * np.pi)
        # assert (angle >= 0 and angle <= 2 * np.pi)
        # angle_per_class = 2 * np.pi / float(NUM_HEADING_BIN)
        # shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        # heading_class = int(shifted_angle / angle_per_class)
        # heading_residual = shifted_angle - (heading_class * angle_per_class + angle_per_class / 2)

        # size_class, size_residual = size2class(self.size_list[index], self.type_list[index])
        # def size2class(size, type_name):
        #     ''' Convert 3D bounding box size to template class and residuals.
        #         todo (rqi): support multiple size clusters per type.
        #         Input:
        #             size: numpy array of shape (3,) for (l,w,h)
        #             type_name: string
        #         Output:
        #             size_class: int scalar
        #             size_residual: numpy array of shape (3,)
        #         '''
        #     size_class = g_type2class[type_name]
        #     size_residual = size - g_type_mean_size[type_name]

        # angle_class, angle_residual = angle2class(heading_angle, NUM_HEADING_BIN)
        # def angle2class(angle, num_class):
        #     ''' Convert continuous angle to discrete class and residual.
        #     Input:
        #         angle: rad scalar, from 0-2pi (or -pi~pi), class center at
        #             0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        #         num_class: int scalar, number of classes N
        #     Output:
        #         class_id, int, among 0,1,...,N-1
        #         residual_angle: float, a number such that
        #             class*(2pi/N) + residual_angle = angle
        #     '''
        #     angle = angle % (2 * np.pi)
        #     assert (angle >= 0 and angle <= 2 * np.pi)
        #     angle_per_class = 2 * np.pi / float(num_class)
        #     shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        #     class_id = int(shifted_angle / angle_per_class)
        #     residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        #     return class_id, residual_angle

        target_dict = {
            'cls_label': cls_label,
            'point_label': point_label,
            'center_label': center_label,
            'head_cls': heading_cls_label,
            'head_res': heading_residual,
            'size_cls': size_cls_label,
            'size_res': size_residual
        }
        return target_dict

    def assign_proposal_target2(self, batch_data, n_sample=2, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        poses = batch_data['pos']
        gt_boxes = batch_data['gt_boxes'][:, :, :7]
        print('assign proposal target:')
        print(poses.shape, gt_boxes.shape)
        label = -torch.ones([len(poses)], dtype=torch.int32)
        # max_ious, gt_argmax_ious = self._calc_ious(poses, gt_boxes)
        # label[max_ious < neg_iou_thresh] = 0
        # label[gt_argmax_ious] = 1
        # label[max_ious >= pos_iou_thresh] = 1
        for i in range(len(poses)):
            pos = torch.unsqueeze(poses[i], 0)
            # pos = poses[i][np.newaxis, :]
            max_ious, gt_argmax_ious = self._calc_ious(pos, gt_boxes[i])
            if max_ious >= pos_iou_thresh:
                label[i] = 1
            elif max_ious < neg_iou_thresh:
                label[i] = 0

        n_pos = int(pos_ratio * n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1
        n_neg = n_sample - torch.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1
        return label

    def assign_targets(self, batch_dict, proposals, rois):
        '''
        :return:
            batch_target['cls_label']: assigned targets for classification of proposals
            batch_target['point_label']: assigned targets for point segmentation
            batch_target['center_label']: center regression label
            batch_target['hclass']: assigned head label for box regression
            batch_target['hres']: head regression label
            batch_target['sclass']: assigned size label for box regression
            batch_target['sres']: size regression label
        '''
        cls_label = self.assign_proposal_target(batch_dict, proposals)

        point_label = self.assign_seg_target(batch_dict, rois)

        center_label = batch_dict['gt_boxes'][:3]
        boxsize = batch_dict['gt_boxes'][3:6]
        heading = batch_dict['gt_boxes'][-1]

        size_class = 0
        size_residual = boxsize - self.g_type_mean_size['Car']

        angle = heading % (2 * np.pi)
        assert (angle >= 0 and angle <= 2 * np.pi)
        angle_per_class = 2 * np.pi / float(NUM_HEADING_BIN)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        heading_class = int(shifted_angle / angle_per_class)
        heading_residual = shifted_angle - (heading_class * angle_per_class + angle_per_class / 2)

        target_dict = {
            'cls_label': cls_label,
            'point_label': point_label,
            'center_label': center_label,
            'hclass': heading_class,
            'hres': heading_residual,
            'sclass': size_class,
            'size': size_residual
        }
        return target_dict

    def assign_proposal_target(self, batch_data, anchor):

        label = np.empty((len(batch_data),), dtype=np.int32)
        label.fill(-1)
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, batch_data['gt_box'])
        label[max_ious < self.neg_iou_thresh] = 0
        label[gt_argmax_ious] = 1
        label[max_ious >= self.pos_iou_thresh] = 1

        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1
        return label

    def _calc_ious(self, anchor, bbox):
        #bbox = torch.from_numpy(bbox).float().cuda()
        ious = boxes_iou3d_gpu(anchor, bbox)  # (N,K)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(anchor.size()[0]), argmax_ious]  # [1,N]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]  # [1,K]
        gt_argmax_ious = torch.where(ious == gt_max_ious)[0]  # K
        return max_ious, gt_argmax_ious

    def assign_seg_target(self, batch_data):
        batch_size = batch_data['batch_size']
        points = batch_data['points'][:, :3, :].swapaxes(2, 1)
        n_pts = points.shape[1]
        #bs_idx = points[:, 0]
        gt_boxes = batch_data['gt_boxes']
        #point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_cls_labels = points.new_zeros(batch_size * n_pts).long()
        for k in range(batch_size):
            # bs_mask = (bs_idx == k)
            # points_single = points[bs_mask][:, 1:4]
            # point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            points_single = points[k, :, :]
            point_cls_labels_single = point_cls_labels.new_zeros(n_pts).long()
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            fg_flag = box_fg_flag
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.n_classes == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[k*n_pts : (k+1)*n_pts] = point_cls_labels_single
        return point_cls_labels

    def point_cloud_masking(self, data_dict, xyz_only=True):
        '''
        :param pts: bs,c,n in frustum
        :param logits: bs,n,2
        :param xyz_only: bool
        :return:
        '''
        pts = data_dict['points']
        logits = data_dict['seg_logits']
        bs = pts.shape[0]
        n_pts = pts.shape[2]
        # Binary Classification for each point
        mask = logits[:, :, 0] < logits[:, :, 1]  # (bs, n)
        mask = mask.unsqueeze(1).float()  # (bs, 1, n)
        mask_count = mask.sum(2, keepdim=True).repeat(1, 3, 1)  # (bs, 3, 1)
        pts_xyz = pts[:, :3, :]  # (bs,3,n)
        mask_xyz_mean = (mask.repeat(1, 3, 1) * pts_xyz).sum(2, keepdim=True)  # (bs, 3, 1)
        mask_xyz_mean = mask_xyz_mean / torch.clamp(mask_count, min=1)  # (bs, 3, 1)
        mask = mask.view(bs, -1)  # (bs,n)
        pts_xyz_stage1 = pts_xyz - mask_xyz_mean.repeat(1, 1, n_pts)

        if xyz_only:
            pts_stage1 = pts_xyz_stage1
        else:
            pts_features = pts[:, 3:, :]
            pts_stage1 = torch.cat([pts_xyz_stage1, pts_features], dim=-1)
        object_pts, _ = self.gather_object_pts(pts_stage1, mask, self.NUM_OBJECT_POINT)
        # (32,512,3) (32,512)
        object_pts = object_pts.reshape(bs, self.NUM_OBJECT_POINT, -1)
        data_dict.update({'pts': object_pts.float().view(bs, 3, -1)})
        data_dict.update({'center': mask_xyz_mean.squeeze()})
        return data_dict

    def gather_object_pts(self, pts, mask, n_pts):
        '''
        :param pts: (bs,c,1024)
        :param mask: (bs,1024)
        :param n_pts: max number of points of an object
        :return:
            object_pts:(bs,c,n_pts)
            indices:(bs,n_pts)
        '''
        bs = pts.shape[0]
        indices = torch.zeros((bs, n_pts), dtype=torch.int64)  # (bs, 512)
        object_pts = torch.zeros((bs, pts.shape[1], n_pts))

        for i in range(bs):
            pos_indices = torch.where(mask[i, :] > 0.5)[0]  # (653,)
            if len(pos_indices) > 0:
                if len(pos_indices) > n_pts:
                    choice = np.random.choice(len(pos_indices),
                                              n_pts, replace=False)
                else:
                    choice = np.random.choice(len(pos_indices),
                                              n_pts - len(pos_indices), replace=True)
                    choice = np.concatenate(
                        (np.arange(len(pos_indices)), choice))
                np.random.shuffle(choice)  # (512,)
                indices[i, :] = pos_indices[choice]
                object_pts[i, :, :] = pts[i, :, indices[i, :]]
            ###else?
        return object_pts, indices

    def parse_output_to_tensors(self, box_pred):
        """
        :param box_pred: (bs,59)
        :return:
            center_boxnet:(bs,3)
            heading_scores:(bs,12)
            heading_residuals_normalized:(bs,12),-1 to 1
            heading_residuals:(bs,12)
            size_scores:(bs,8)
            size_residuals_normalized:(bs,8)
            size_residuals:(bs,8)
        """
        bs = box_pred.shape[0]
        # center
        center_boxnet = box_pred[:, :3]  # 0:3

        # heading
        c = 3
        heading_scores = box_pred[:, c:c + NUM_HEADING_BIN]  # 3:3+12
        c += NUM_HEADING_BIN
        heading_residuals_normalized = box_pred[:, c:c + NUM_HEADING_BIN]  # 3+12 : 3+2*12
        heading_residuals = heading_residuals_normalized * (np.pi / NUM_HEADING_BIN)
        c += NUM_HEADING_BIN

        # size
        size_scores = box_pred[:, c:c + NUM_SIZE_CLUSTER]  # 3+2*12 : 3+2*12+8
        c += NUM_SIZE_CLUSTER
        size_residuals_normalized = box_pred[:, c:c + 3 * NUM_SIZE_CLUSTER].contiguous()  # [32,24] 3+2*12+8 : 3+2*12+4*8
        size_residuals_normalized = size_residuals_normalized.view(bs, NUM_SIZE_CLUSTER, 3)  # [32,8,3]
        size_residuals = size_residuals_normalized * \
                         torch.from_numpy(self.g_mean_size_arr).unsqueeze(0).repeat(bs, 1, 1).cuda()
        return center_boxnet, heading_scores, heading_residuals_normalized, heading_residuals, \
               size_scores, size_residuals_normalized, size_residuals

    def get_box3d_corners_helper(self, centers, headings, sizes):
        """ Input: (N,3), (N,), (N,3), Output: (N,8,3) """
        # print '-----', centers
        N = centers.shape[0]
        l = sizes[:, 0].view(N, 1)
        w = sizes[:, 1].view(N, 1)
        h = sizes[:, 2].view(N, 1)
        # print l,w,h
        x_corners = torch.cat([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dim=1)  # (N,8)
        y_corners = torch.cat([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], dim=1)  # (N,8)
        z_corners = torch.cat([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dim=1)  # (N,8)
        corners = torch.cat([x_corners.view(N, 1, 8), y_corners.view(N, 1, 8),
                             z_corners.view(N, 1, 8)], dim=1)  # (N,3,8)

        ###ipdb.set_trace()
        # print x_corners, y_corners, z_corners
        c = torch.cos(headings).cuda()
        s = torch.sin(headings).cuda()
        ones = torch.ones([N], dtype=torch.float32).cuda()
        zeros = torch.zeros([N], dtype=torch.float32).cuda()
        row1 = torch.stack([c, zeros, s], dim=1)  # (N,3)
        row2 = torch.stack([zeros, ones, zeros], dim=1)
        row3 = torch.stack([-s, zeros, c], dim=1)
        mat = torch.cat([row1.view(N, 1, 3), row2.view(N, 1, 3), row3.view(N, 1, 3)], axis=1)  # (N,3,3)
        # print row1, row2, row3, R, N
        corners_3d = torch.bmm(mat, corners)  # (N,3,8)
        corners_3d += centers.view(N, 3, 1).repeat(1, 1, 8)  # (N,3,8)
        corners_3d = torch.transpose(corners_3d, 1, 2)  # (N,8,3)
        return corners_3d

    def get_box3d_corners(self, center, heading_residuals, size_residuals):
        """
        Inputs:
            center: (bs,3)
            heading_residuals: (bs,NH)
            size_residuals: (bs,NS,3)
        Outputs:
            box3d_corners: (bs,NH,NS,8,3) tensor
        """
        bs = center.shape[0]
        heading_bin_centers = torch.from_numpy(
            np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN)).float()  # (12,) (NH,)
        headings = heading_residuals + heading_bin_centers.view(1, -1).cuda()  # (bs,12)

        mean_sizes = torch.from_numpy(self.g_mean_size_arr).float().view(1, NUM_SIZE_CLUSTER, 3).cuda() \
                     + size_residuals.cuda()  # (1,8,3)+(bs,8,3) = (bs,8,3)
        sizes = mean_sizes + size_residuals  # (bs,8,3)
        sizes = sizes.view(bs, 1, NUM_SIZE_CLUSTER, 3) \
            .repeat(1, NUM_HEADING_BIN, 1, 1).float()  # (B,12,8,3)
        headings = headings.view(bs, NUM_HEADING_BIN, 1).repeat(1, 1, NUM_SIZE_CLUSTER)  # (bs,12,8)
        centers = center.view(bs, 1, 1, 3).repeat(1, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1)  # (bs,12,8,3)
        N = bs * NUM_HEADING_BIN * NUM_SIZE_CLUSTER
        ###ipdb.set_trace()
        corners_3d = self.get_box3d_corners_helper(centers.view(N, 3), headings.view(N),
                                                   sizes.view(N, 3))
        ###ipdb.set_trace()
        return corners_3d.view(bs, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3)  # [32, 12, 8, 8, 3]

    def huber_loss(self, error, delta=1.0):  # (32,), ()
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=delta)
        linear = (abs_error - quadratic)
        losses = 0.5 * quadratic ** 2 + delta * linear
        return torch.mean(losses)

    # @torch.no_grad()
    # def generate_proposals(self, batch_dict, model_cfg):
    #     [dx, dy, dz] = self.model_cfg.ANCHOR_GENERATOR_CONFIG[0]['anchor_sizes'][0]  # only car target
    #     batch_size = batch_dict['batch_size']
    #     pc_size = int(batch_dict['points'].shape[0] / batch_size)
    #     batch_proposal_pose = batch_dict['points'].new_zeros((18*batch_dict['points'].shape[0], 7))
    #     batch_proposal_pts = batch_dict['points'].new_zeros((18*batch_dict['points'].shape[0], 128, 4))
    #     batch_frame_id = batch_dict['points'].new_zeros((18*batch_dict['points'].shape[0])).long()
    #     for m in range(batch_size):
    #         pts = batch_dict['points'][m * pc_size:(m + 1) * pc_size]
    #         for n in range(pts.size()[0]):
    #             xc, yc, zc = pts[n, 1:4]
    #             centers_xy = torch.tensor([
    #                 [xc, yc], [xc + dx / 4, yc], [xc - dx / 4, yc], [xc, yc + dy / 4], [xc, yc - dy / 4],
    #                 [xc + dx / 4, yc + dy / 4], [xc + dx / 4, yc - dy / 4], [xc - dx / 4, yc + dy / 4],
    #                 [xc - dx / 4, yc - dy / 4],
    #                 [xc, yc], [xc + dy / 4, yc], [xc - dy / 4, yc], [xc, yc + dx / 4], [xc, yc + dx / 4],
    #                 [xc + dy / 4, yc + dx / 4], [xc + dy / 4, yc - dx / 4], [xc - dy / 4, yc + dx / 4],
    #                 [xc - dy / 4, yc - dx / 4]
    #             ])
    #             poses = batch_dict['points'].new_zeros((18, 7))  #np.zeros((18, 7))
    #             poses[:, :2] = centers_xy
    #             poses[:, 2:6] = torch.tensor([zc, dx, dy, dz])
    #             poses[9:, -1] = np.pi/2
    #             corners3d = boxes_to_corners_3d(poses)
    #             #poses = torch.from_numpy(poses)
    #             for k in range(len(poses)):
    #                 flag = in_hull(pts[:, 1:4].cpu(), corners3d[k].cpu())
    #                 idx = [i for i, x in enumerate(flag) if x == 1]
    #                 idx_sample = np.random.choice(idx, 128, replace=True)  # move to model_cfg later
    #                 batch_proposal_pts[m*pc_size + n*18 + k, :, :] = pts[idx_sample, 1:5]
    #                 batch_proposal_pose[m*pc_size + n*18 + k, :] = poses[k]
    #                 batch_frame_id[m*pc_size + n*18 + k] = int(batch_dict['frame_id'][m])
    #
    #     batch_proposal_pts = batch_proposal_pts.permute(0, 2, 1).contiguous()
    #     proposals = {
    #         'frame_id': batch_frame_id,
    #         'pos': batch_proposal_pose,
    #         'pts': batch_proposal_pts
    #     }
    #     return proposals

    @torch.no_grad()
    def get_rois(self, proposals, features, n_pre_nms=10000, n_post_nms=2000):

        proposals.update(features)
        indices = [index for index, value in enumerate(proposals['cls_pred'][:, 1]) if value == 1]
        indices = [0, 1]
        print(indices)
        rois = {}
        for key, value in proposals.items():
            if key == 'batch_size':
                pass
            else:
                rois[key] = value[indices]
        rois.update({'batch_size': len(indices)})
        print(f'get number of ROIs: %d' % rois['batch_size'])

        # rois = [dict(item, **{'feature': features[index]}) for index, item in enumerate(proposals)]
        # order = proposals['cls_pred'].ravel().argsort()[::-1]
        # order = order[:n_pre_nms]
        # rois = rois[order, :]
        # keep = class_agnostic_nms(rois)
        # keep = keep[:n_post_nms]
        # rois = rois[keep]

        return rois
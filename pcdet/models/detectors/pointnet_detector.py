import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pcdet.utils.box_utils import in_hull, boxes_to_corners_3d

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8 # one cluster for each type


def make_fc_layers(fc_cfg, input_channels, output_channels):
    fc_layers = []
    c_in = input_channels
    for k in range(0, fc_cfg.__len__()):
        fc_layers.extend([
            nn.Linear(c_in, fc_cfg[k], bias=False),
            nn.BatchNorm1d(fc_cfg[k]),
            nn.ReLU(),
        ])
        c_in = fc_cfg[k]
    fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
    return nn.Sequential(*fc_layers)


class PointNetv1(nn.Module):
    def __init__(self, num_class=3, input_channels=4):
        super(PointNetv1, self).__init__()
        self.num_class = num_class

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
        self.cls_layers = self.make_fc_layers(input_channels=input_channels, output_channels=num_class + 1)

    def forward(self, batch_data):  # bs,4,n
        bs = batch_data.size()[0]
        n_pts = batch_data.size()[2]

        out1 = F.relu(self.bn1(self.conv1(batch_data)))  # bs,64,n
        out2 = F.relu(self.bn2(self.conv2(out1)))  # bs,64,n
        out3 = F.relu(self.bn3(self.conv3(out2)))  # bs,64,n
        out4 = F.relu(self.bn4(self.conv4(out3)))  # bs,128,n
        out5 = F.relu(self.bn5(self.conv5(out4)))  # bs,1024,n
        global_feat = torch.max(out5, 2, keepdim=True)[0]  # bs,1024,1
        cls_score = self.cls_layers(global_feat)

        softmax = nn.Softmax(dim=1)
        cls_pred = softmax(cls_score)
        cls_pred = torch.max(cls_pred, 1)
        one_hot_vec = torch.zeros(self.num_class)
        one_hot_vec[cls_pred] = 1

        feature_dict = {
            'local_feat': out2,
            'global_feat': global_feat,
            'cls_pred': one_hot_vec,
            'cls_score': cls_score
        }
        return feature_dict


class PointSeg(nn.Module):
    def __init__(self, num_class=3, input_channels=4):
        super(PointSeg, self).__init__()

        self.dconv1 = nn.Conv1d(1088 + num_class, 512, 1)
        self.dconv2 = nn.Conv1d(512, 256, 1)
        self.dconv3 = nn.Conv1d(256, 128, 1)
        self.dconv4 = nn.Conv1d(128, 128, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.dconv5 = nn.Conv1d(128, 2, 1)
        self.dbn1 = nn.BatchNorm1d(512)
        self.dbn2 = nn.BatchNorm1d(256)
        self.dbn3 = nn.BatchNorm1d(128)
        self.dbn4 = nn.BatchNorm1d(128)
        self.cls_layers = self.make_fc_layers(input_channels=input_channels, output_channels=num_class + 1)

    def forward(self, data_dict):  # bs,4,n
        bs = data_dict['pts'].size()[0]
        n_pts = data_dict['pts'].size()[2]

        expand_one_hot_vec = data_dict['cls_score'].view(bs, -1, 1)  # bs,3,1
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
        return seg_pred


class CenterRegNet(nn.Module):
    def __init__(self, n_classes=3):
        super(CenterRegNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        # self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(256 + n_classes, 256)
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
        bs = data_dict['pts'].size()[0]
        x = F.relu(self.bn1(self.conv1(data_dict['pts'])))  # bs,128,n
        x = F.relu(self.bn2(self.conv2(x)))  # bs,128,n
        x = F.relu(self.bn3(self.conv3(x)))  # bs,256,n
        x = torch.max(x, 2)[0]  # bs,256
        expand_one_hot_vec = data_dict['cls_score'].vw(bs, -1)  # bs,3
        x = torch.cat([x, expand_one_hot_vec], 1)  # bs,259
        x = F.relu(self.fcbn1(self.fc1(x)))  # bs,256
        x = F.relu(self.fcbn2(self.fc2(x)))  # bs,128
        x = self.fc3(x)  # bs,
        ###if np.isnan(x.cpu().detach().numpy()).any():
        ###    ipdb.set_trace()
        return x


class BoxRegNet(nn.Module):
    def __init__(self, n_classes=3):
        '''Amodal 3D Box Estimation Pointnet
        :param n_classes:3
        :param one_hot_vec:[bs,n_classes]
        '''
        super(BoxRegNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.n_classes = n_classes

        self.fc1 = nn.Linear(512 + n_classes, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)

    def forward(self, data_dict):  # bs,3,m
        '''
        :param pts: [bs,3,m]: x,y,z after InstanceSeg
        :return: box_pred: [bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4]
            including box centers, heading bin class scores and residuals,
            and size cluster scores and residuals
        '''
        bs = data_dict['pts'].size()[0]

        out1 = F.relu(self.bn1(self.conv1(data_dict['pts'])))  # bs,128,n
        out2 = F.relu(self.bn2(self.conv2(out1)))  # bs,128,n
        out3 = F.relu(self.bn3(self.conv3(out2)))  # bs,256,n
        out4 = F.relu(self.bn4(self.conv4(out3)))  # bs,512,n
        global_feat = torch.max(out4, 2, keepdim=False)[0]  # bs,512

        expand_one_hot_vec = data_dict['cls_score'].view(bs, -1)  # bs,3
        expand_global_feat = torch.cat([data_dict['global_feat'], expand_one_hot_vec], 1)  # bs,515

        x = F.relu(self.fcbn1(self.fc1(expand_global_feat)))  # bs,512
        x = F.relu(self.fcbn2(self.fc2(x)))  # bs,256
        box_pred = self.fc3(x)  # bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4
        return box_pred


class PointNetDetector(nn.Module):
    def __init__(self, model_cfg, dataset, n_classes=3, n_channel=4):
        super(PointNetDetector, self).__init__()
        self.model_cfg = model_cfg
        self.n_classes = n_classes
        self.dataset = dataset
        self.PointCls = PointNetv1(n_classes=3, n_channel=n_channel)
        self.PointSeg = PointSeg(n_classes=3, n_channel=n_channel)
        self.CenterReg = CenterRegNet(n_classes=3)
        self.BoxReg = BoxRegNet(n_classes=3)
        self.NUM_OBJECT_POINT = 512

    def forward(self, batch_dict):  # bs,4,n

        proposals = self.generate_proposals(batch_dict)

        batch_target = self.generate_targets(batch_dict, proposals)

        feature_dict = self.PointNetv1(proposals['pts'])

        #ret_dict = {'target_cls_preds': feature_dict['cls_pred']}
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            #ret_dict['target_cls_labels'] = targets_dict['target_cls_labels']
            cls_loss = F.cross_entropy(feature_dict['cls_score'], targets_dict['target_cls_labels'])

        proposals = self.pop_proposals(proposals, feature_dict)

        # 3D Instance Segmentation PointNet
        logits = self.PointSeg(proposals['pts'], feature_dict)  # bs,n,2

        # Mask Point Centroid
        object_pts_xyz, mask_xyz_mean, mask = self.point_cloud_masking(proposals['pts'], logits)  ###logits.detach()

        # T-Net
        object_pts_xyz = object_pts_xyz.cuda()
        center_delta = self.CenterReg(object_pts_xyz)  # (32,3)
        stage1_center = center_delta + mask_xyz_mean  # (32,3)

        # if (np.isnan(stage1_center.cpu().detach().numpy()).any()):
        #     ipdb.set_trace()
        object_pts_xyz_new = object_pts_xyz - \
                             center_delta.view(center_delta.shape[0], -1, 1).repeat(1, 1, object_pts_xyz.shape[-1])

        # 3D Box Estimation
        box_pred = self.BoxReg(object_pts_xyz_new)  # (32, 59)

        center_boxnet, heading_scores, heading_residuals_normalized, heading_residuals, \
        size_scores, size_residuals_normalized, size_residuals = \
            self.parse_output_to_tensors(box_pred, logits, mask, stage1_center)

        center = center_boxnet + stage1_center  # bs,3

        if self.training:
            loss, tb_dict, disp_dict = self.get_loss(feature_dict, logits, batch_target['label'], center,
                                                     batch_target['center'], stage1_center, heading_scores,
                                                     heading_residuals_normalized, heading_residuals,
                                                     batch_target['hclass'], batch_target['hres'], size_scores,
                                                     size_residuals_normalized, size_residuals, batch_target['sclass'],
                                                     batch_target['sres'])

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_loss(self, feature_dict, logits, mask_label, \
                 center, center_label, stage1_center, \
                 heading_scores, heading_residual_normalized, heading_residual, \
                 heading_class_label, heading_residual_label, \
                 size_scores, size_residual_normalized, size_residual,
                 size_class_label, size_residual_label,
                 corner_loss_weight=10.0, box_loss_weight=1.0):
        '''
        1.PointSeg
        logits: torch.Size([32, 1024, 2]) torch.float32
        mask_label: [32, 1024]
        2.Center
        center: torch.Size([32, 3]) torch.float32
        stage1_center: torch.Size([32, 3]) torch.float32
        center_label:[32,3]
        3.Heading
        heading_scores: torch.Size([32, 12]) torch.float32
        heading_residual_snormalized: torch.Size([32, 12]) torch.float32
        heading_residual: torch.Size([32, 12]) torch.float32
        heading_class_label:(32)
        heading_residual_label:(32)
        4.Size
        size_scores: torch.Size([32, 8]) torch.float32
        size_residual_normalized: torch.Size([32, 8, 3]) torch.float32
        size_residual: torch.Size([32, 8, 3]) torch.float32
        size_class_label:(32)
        size_residual_label:(32,3)
        5.Corner
        6.Weight
        corner_loss_weight: float scalar
        box_loss_weight: float scalar
        '''
        g_type2class = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3,
                        'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7}
        g_class2type = {g_type2class[t]: t for t in g_type2class}
        g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        g_type_mean_size = {'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
                            'Van': np.array([5.06763659, 1.9007158, 2.20532825]),
                            'Truck': np.array([10.13586957, 2.58549199, 3.2520595]),
                            'Pedestrian': np.array([0.84422524, 0.66068622, 1.76255119]),
                            'Person_sitting': np.array([0.80057803, 0.5983815, 1.27450867]),
                            'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127]),
                            'Tram': np.array([16.17150617, 2.53246914, 3.53079012]),
                            'Misc': np.array([3.64300781, 1.54298177, 1.92320313])}
        g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs
        for i in range(NUM_SIZE_CLUSTER):
            g_mean_size_arr[i, :] = g_type_mean_size[g_class2type[i]]

        bs = logits.shape[0]
        # 3D Proposal Classification Loss
        cls_loss = F.cross_entropy(feature_dict['cls_score'], cls_target)

        # 3D Instance Segmentation PointNet Loss
        logits = F.log_softmax(logits.view(-1, 2), dim=1)  # torch.Size([32768, 2])
        mask_label = mask_label.view(-1).long()  # torch.Size([32768])
        mask_loss = F.nll_loss(logits, mask_label)  # tensor(0.6361, grad_fn=<NllLossBackward>)

        # Center Regression Loss
        center_dist = torch.norm(center - center_label, dim=1)  # (32,)
        center_loss = self.huber_loss(center_dist, delta=2.0)

        stage1_center_dist = torch.norm(center - stage1_center, dim=1)  # (32,)
        stage1_center_loss = self.huber_loss(stage1_center_dist, delta=1.0)

        # Heading Loss
        heading_class_loss = F.nll_loss(F.log_softmax(heading_scores, dim=1), \
                                        heading_class_label.long())  # tensor(2.4505, grad_fn=<NllLossBackward>)
        hcls_onehot = torch.eye(NUM_HEADING_BIN)[heading_class_label.long()].cuda()  # 32,12
        heading_residual_normalized_label = \
            heading_residual_label / (np.pi / NUM_HEADING_BIN)  # 32,
        heading_residual_normalized_dist = torch.sum( \
            heading_residual_normalized * hcls_onehot.float(), dim=1)  # 32,
        ### Only compute reg loss on gt label
        heading_residual_normalized_loss = \
            self.huber_loss(heading_residual_normalized_dist -
                            heading_residual_normalized_label, delta=1.0)  ###fix,2020.1.14
        # Size loss
        size_class_loss = F.nll_loss(F.log_softmax(size_scores, dim=1), \
                                     size_class_label.long())  # tensor(2.0240, grad_fn=<NllLossBackward>)

        scls_onehot = torch.eye(NUM_SIZE_CLUSTER)[size_class_label.long()].cuda()  # 32,8
        scls_onehot_repeat = scls_onehot.view(-1, NUM_SIZE_CLUSTER, 1).repeat(1, 1, 3)  # 32,8,3
        predicted_size_residual_normalized_dist = torch.sum( \
            size_residual_normalized * scls_onehot_repeat.cuda(), dim=1)  # 32,3
        mean_size_arr_expand = torch.from_numpy(g_mean_size_arr).float().cuda() \
            .view(1, NUM_SIZE_CLUSTER, 3)  # 1,8,3
        mean_size_label = torch.sum(scls_onehot_repeat * mean_size_arr_expand, dim=1)  # 32,3
        size_residual_label_normalized = size_residual_label / mean_size_label.cuda()

        size_normalized_dist = torch.norm(size_residual_label_normalized - \
                                          predicted_size_residual_normalized_dist, dim=1)  # 32
        size_residual_normalized_loss = self.huber_loss(size_normalized_dist,
                                                        delta=1.0)  # tensor(11.2784, grad_fn=<MeanBackward0>)

        # Corner Loss
        corners_3d = self.get_box3d_corners(center, heading_residual,
                                            size_residual).cuda()  # (bs,NH,NS,8,3)(32, 12, 8, 8, 3)
        gt_mask = hcls_onehot.view(bs, NUM_HEADING_BIN, 1).repeat(1, 1, NUM_SIZE_CLUSTER) * \
                  scls_onehot.view(bs, 1, NUM_SIZE_CLUSTER).repeat(1, NUM_HEADING_BIN, 1)  # (bs,NH=12,NS=8)
        corners_3d_pred = torch.sum(gt_mask.view(bs, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1, 1) \
                                    .float().cuda() * corners_3d, dim=[1, 2])  # (bs,8,3)
        heading_bin_centers = torch.from_numpy(
            np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN)).float().cuda()  # (NH,)
        heading_label = heading_residual_label.view(bs, 1) + heading_bin_centers.view(1,
                                                                                      NUM_HEADING_BIN)  # (bs,1)+(1,NH)=(bs,NH)

        heading_label = torch.sum(hcls_onehot.float() * heading_label, 1)
        mean_sizes = torch.from_numpy(g_mean_size_arr).float().view(1, NUM_SIZE_CLUSTER, 3).cuda()  # (1,NS,3)
        size_label = mean_sizes + size_residual_label.view(bs, 1, 3)  # (1,NS,3)+(bs,1,3)=(bs,NS,3)
        size_label = torch.sum(scls_onehot.view(bs, NUM_SIZE_CLUSTER, 1).float() * size_label, axis=[1])  # (B,3)

        corners_3d_gt = self.get_box3d_corners_helper( \
            center_label, heading_label, size_label)  # (B,8,3)
        corners_3d_gt_flip = self.get_box3d_corners_helper( \
            center_label, heading_label + np.pi, size_label)  # (B,8,3)

        corners_dist = torch.min(torch.norm(corners_3d_pred - corners_3d_gt, dim=-1),
                                 torch.norm(corners_3d_pred - corners_3d_gt_flip, dim=-1))
        corners_loss = self.huber_loss(corners_dist, delta=1.0)

        # Weighted sum of all losses
        total_loss = mask_loss + box_loss_weight * (center_loss + \
                                                    heading_class_loss + size_class_loss + \
                                                    heading_residual_normalized_loss * 20 + \
                                                    size_residual_normalized_loss * 20 + \
                                                    stage1_center_loss + \
                                                    corner_loss_weight * corners_loss)

        losses = {
            'total_loss': total_loss,
            'mask_loss': mask_loss,
            'center_loss': box_loss_weight * center_loss,
            'heading_class_loss': box_loss_weight * heading_class_loss,
            'size_class_loss': box_loss_weight * size_class_loss,
            'heading_residual_normalized_loss': box_loss_weight * heading_residual_normalized_loss * 20,
            'size_residual_normalized_loss': box_loss_weight * size_residual_normalized_loss * 20,
            'stage1_center_loss': box_loss_weight * size_residual_normalized_loss * 20,
            'corners_loss': box_loss_weight * corners_loss * corner_loss_weight,
        }
        return losses

    def generate_proposals(self, batch_dict):
        dx, dy, dz = self.model_cfg.ANCHOR_GENERATOR_CONFIG[0]['anchor_sizes']
        batch_proposal_poses = []
        batch_proposal_points = []
        batch_frame_ids = []
        for index, data in enumerate(batch_dict):
            poses = []
            for pt in data['points']:
                pos = []
                xc, yc, zc = pt
                pos.append([xc, yc, zc, dx, dy, dz])
                pos.append([xc + dx / 4, yc, zc, dx, dy, dz])
                pos.append([xc - dx / 4, yc, zc, dx, dy, dz])
                pos.append([xc, yc + dy / 4, zc, dx, dy, dz])
                pos.append([xc, yc - dy / 4, zc, dx, dy, dz])
                pos.append([xc + dx / 4, yc + dy / 4, zc, dx, dy, dz])
                pos.append([xc + dx / 4, yc - dy / 4, zc, dx, dy, dz])
                pos.append([xc - dx / 4, yc + dy / 4, zc, dx, dy, dz])
                pos.append([xc - dx / 4, yc - dy / 4, zc, dx, dy, dz])
                pos_horizon = [[*pr, 0] for pr in pos]
                pos_vertica = [[*pr, np.pi / 2] for pr in pos]
                poses.append(pos_horizon + pos_vertica)
            frame_ids = []
            indices = []
            corners3d = boxes_to_corners_3d(poses)
            for k in range(len(poses)):
                flag = in_hull(data['points'][:, 0:3], corners3d[k])
                indice = [i for i, x in enumerate(flag) if x == 1]
                indices.extend(indice)
                frame_ids.append(data['frame_id'])
            points = data['points'][indices]

            batch_proposal_poses.append(*poses)
            batch_proposal_points.append(*points)
            batch_frame_ids.append(*frame_ids)

        proposals = {'frame_id': batch_frame_ids, 'pos': batch_proposal_poses, 'pts': batch_proposal_points}
        return proposals

    def point_cloud_masking(self, pts, logits, xyz_only=True):
        '''
        :param pts: bs,c,n in frustum
        :param logits: bs,n,2
        :param xyz_only: bool
        :return:
        '''
        bs = pts.shape[0]
        n_pts = pts.shape[2]
        # Binary Classification for each point
        mask = logits[:, :, 0] < logits[:, :, 1]  # (bs, n)
        mask = mask.unsqueeze(1).float()# (bs, 1, n)
        mask_count = mask.sum(2,keepdim=True).repeat(1, 3, 1)  # (bs, 3, 1)
        pts_xyz = pts[:, :3, :]  # (bs,3,n)
        mask_xyz_mean = (mask.repeat(1, 3, 1) * pts_xyz).sum(2,keepdim=True)  # (bs, 3, 1)
        mask_xyz_mean = mask_xyz_mean / torch.clamp(mask_count,min=1)  # (bs, 3, 1)
        mask = mask.view(bs,-1)  # (bs,n)
        pts_xyz_stage1 = pts_xyz - mask_xyz_mean.repeat(1, 1, n_pts)

        if xyz_only:
            pts_stage1 = pts_xyz_stage1
        else:
            pts_features = pts[:, 3:, :]
            pts_stage1 = torch.cat([pts_xyz_stage1, pts_features], dim=-1)
        object_pts, _ = self.gather_object_pts(pts_stage1, mask, self.NUM_OBJECT_POINT)
        #(32,512,3) (32,512)
        object_pts = object_pts.reshape(bs, self.NUM_OBJECT_POINT, -1)
        object_pts = object_pts.float().view(bs,3,-1)
        return object_pts, mask_xyz_mean.squeeze(), mask

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
                object_pts[i,:,:] = pts[i,:,indices[i,:]]
            ###else?
        return object_pts, indices

    def parse_output_to_tensors(self, box_pred, logits, mask, stage1_center):
        '''
        :param box_pred: (bs,59)
        :param logits: (bs,1024,2)
        :param mask: (bs,1024)
        :param stage1_center: (bs,3)
        :return:
            center_boxnet:(bs,3)
            heading_scores:(bs,12)
            heading_residuals_normalized:(bs,12),-1 to 1
            heading_residuals:(bs,12)
            size_scores:(bs,8)
            size_residuals_normalized:(bs,8)
            size_residuals:(bs,8)
        '''
        bs = box_pred.shape[0]
        # center
        center_boxnet = box_pred[:, :3]#0:3
        c = 3

        # heading
        heading_scores = box_pred[:, c:c + NUM_HEADING_BIN]#3:3+12
        c += NUM_HEADING_BIN
        heading_residuals_normalized = \
            box_pred[:, c:c + NUM_HEADING_BIN]#3+12 : 3+2*12
        heading_residuals = \
            heading_residuals_normalized * (np.pi / NUM_HEADING_BIN)
        c += NUM_HEADING_BIN

        # size
        size_scores = box_pred[:, c:c + NUM_SIZE_CLUSTER]#3+2*12 : 3+2*12+8
        c += NUM_SIZE_CLUSTER
        size_residuals_normalized = \
            box_pred[:, c:c + 3 * NUM_SIZE_CLUSTER].contiguous() #[32,24] 3+2*12+8 : 3+2*12+4*8
        size_residuals_normalized = \
            size_residuals_normalized.view(bs,NUM_SIZE_CLUSTER,3)#[32,8,3]
        size_residuals = size_residuals_normalized * \
                         torch.from_numpy(g_mean_size_arr).unsqueeze(0).repeat(bs,1,1).cuda()
        return center_boxnet,\
                heading_scores, heading_residuals_normalized, heading_residuals,\
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
        corners = torch.cat([x_corners.view(N, 1, 8), y_corners.view(N, 1, 8), \
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
        R = torch.cat([row1.view(N, 1, 3), row2.view(N, 1, 3), \
                       row3.view(N, 1, 3)], axis=1)  # (N,3,3)
        # print row1, row2, row3, R, N
        corners_3d = torch.bmm(R, corners)  # (N,3,8)
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
        heading_bin_centers = torch.from_numpy( \
            np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN)).float()  # (12,) (NH,)
        headings = heading_residuals + heading_bin_centers.view(1, -1).cuda()  # (bs,12)

        mean_sizes = torch.from_numpy(g_mean_size_arr).float().view(1, NUM_SIZE_CLUSTER, 3).cuda() \
                     + size_residuals.cuda()  # (1,8,3)+(bs,8,3) = (bs,8,3)
        sizes = mean_sizes + size_residuals  # (bs,8,3)
        sizes = sizes.view(bs, 1, NUM_SIZE_CLUSTER, 3) \
            .repeat(1, NUM_HEADING_BIN, 1, 1).float()  # (B,12,8,3)
        headings = headings.view(bs, NUM_HEADING_BIN, 1).repeat(1, 1, NUM_SIZE_CLUSTER)  # (bs,12,8)
        centers = center.view(bs, 1, 1, 3).repeat(1, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1)  # (bs,12,8,3)
        N = bs * NUM_HEADING_BIN * NUM_SIZE_CLUSTER
        ###ipdb.set_trace()
        corners_3d = self.get_box3d_corners_helper(centers.view(N, 3), headings.view(N), \
                                              sizes.view(N, 3))
        ###ipdb.set_trace()
        return corners_3d.view(bs, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3)  # [32, 12, 8, 8, 3]

    def huber_loss(self, error, delta=1.0):  # (32,), ()
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=delta)
        linear = (abs_error - quadratic)
        losses = 0.5 * quadratic ** 2 + delta * linear
        return torch.mean(losses)

    @torch.no_grad()
    def pop_proposals(self, prop_list):
        prop_filtered = []
        for prop in prop_list:
            if torch.argmax(prop['cls_pred']) == 1:
                prop_filtered.append(prop)
        return prop_filtered

    def generate_targets(self, batch_dict, proposals):
        target_dict = {'label': [],
                       'center': [],
                       'hclass': [],
                       'hres': [],
                       'sclass': [],
                       'sres': []}

        return target_dict
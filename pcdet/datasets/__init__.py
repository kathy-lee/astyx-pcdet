import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .nuscenes.nuscenes_dataset import NuScenesDataset
from .astyx.astyx_dataset import AstyxDataset

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'NuScenesDataset': NuScenesDataset,
    'AstyxDataset': AstyxDataset
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler


def build_proposal_dataloader(anchor_cfg, dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    pc_dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )
    dataset = generate_proposals(anchor_cfg, pc_dataset)
    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler

def generate_proposals(anchor_cfg, dataset):
    import numpy as np
    from pcdet.utils.box_utils import in_hull, boxes_to_corners_3d

    [dx, dy, dz] = anchor_cfg[0]['anchor_sizes'][0]  # only car target
    n_pt = dataset[0]['points'].shape[0]
    n_pc = len(dataset)
    batch_proposal_pose = np.empty([18 * n_pt * n_pc, 7])
    batch_proposal_pts = np.empty([18 * n_pt * n_pc, 128, 4])
    batch_frame_id = np.empty([18 * n_pt * n_pc], dtype=int)
    for m in range(n_pc):
        print(f'genreate proposals for %d th pc(frame id %s)' % (m, dataset[m]['frame_id']))
        pts = dataset[m]['points']
        for n in range(n_pt):
            xc, yc, zc = pts[n, 1:4]
            centers_xy = np.array([
                [xc, yc], [xc + dx / 4, yc], [xc - dx / 4, yc], [xc, yc + dy / 4], [xc, yc - dy / 4],
                [xc + dx / 4, yc + dy / 4], [xc + dx / 4, yc - dy / 4], [xc - dx / 4, yc + dy / 4],
                [xc - dx / 4, yc - dy / 4],
                [xc, yc], [xc + dy / 4, yc], [xc - dy / 4, yc], [xc, yc + dx / 4], [xc, yc + dx / 4],
                [xc + dy / 4, yc + dx / 4], [xc + dy / 4, yc - dx / 4], [xc - dy / 4, yc + dx / 4],
                [xc - dy / 4, yc - dx / 4]
            ])
            poses = np.zeros([18, 7])
            poses[:, :2] = centers_xy
            poses[:, 2:6] = np.array([zc, dx, dy, dz])
            poses[9:, -1] = np.pi / 2
            corners3d = boxes_to_corners_3d(poses)
            for k in range(len(poses)):
                flag = in_hull(pts[:, 1:4], corners3d[k])
                idx = [i for i, x in enumerate(flag) if x == 1]
                idx_sample = np.random.choice(idx, 128, replace=True)  # move to model_cfg later
                batch_proposal_pts[m * n_pt + n * 18 + k, :, :] = pts[idx_sample, 1:5]
                batch_proposal_pose[m * n_pt + n * 18 + k, :] = poses[k]
                batch_frame_id[m * n_pt + n * 18 + k] = int(dataset[m]['frame_id'])

    batch_proposal_pts = batch_proposal_pts.swapaxes(2,1)
    proposals = {
        'frame_id': batch_frame_id,
        'pos': batch_proposal_pose,
        'pts': batch_proposal_pts
    }
    return proposals
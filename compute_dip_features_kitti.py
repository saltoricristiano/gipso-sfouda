import os
import numpy as np
import torch
import open3d as o3d
from models.network import PointNetFeature
import time
from models.lrf import lrf
import glob
import warnings
import argparse
from tqdm import tqdm
import MinkowskiEngine as ME

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--source_path",
                    default="/data/csaltori/SemanticKITTI/data/sequences",
                    type=str,
                    help="Path to dataset")
parser.add_argument("--save_path",
                    default="experiments/dip_features",
                    type=str,
                    help="Path to save")
parser.add_argument("--split",
                    default=1,
                    type=int,
                    help="Num of splits")


patch_size = 256
dim = 32
batch_size = 500
ratio_to_sample = 1.
lrf_kernel = 2.5
voxel_size = 0.3

net = PointNetFeature(dim=dim)
checkpoint = './experiments/dip_model/final_chkpt.pth'
net.load_state_dict(torch.load(checkpoint))
net = net.cuda()
net.eval()


def compute(fpcd, fdip):
    xyzr = np.fromfile(fpcd, dtype=np.float32).reshape(-1, 4)
    xyz = xyzr[:, :3]

    coords = np.floor(xyz / voxel_size)
    _, voxel_idx = ME.utils.sparse_quantize(coords, return_index=True)

    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(xyz)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[voxel_idx.numpy()])

    pcd_pts = np.asarray(pcd.points)
    lrf1 = lrf(pcd=pcd,
               pcd_tree=o3d.geometry.KDTreeFlann(pcd),
               patch_size=patch_size,
               lrf_kernel=lrf_kernel,
               viz=False)

    patches = np.empty((len(pcd_pts), 3, patch_size))

    for i in range(len(pcd_pts)):
        try:
            lrf1_pts, _, _ = lrf1.get(pcd_pts[i])
            patches[i] = lrf1_pts.T
        except:
            patches[i] = np.ones((3, 256))

    pcd_desc = np.empty((patches.shape[0], dim))

    for b in range(int(np.ceil(patches.shape[0] / batch_size))):

        i_start = b * batch_size
        i_end = (b + 1) * batch_size
        if i_end > int(ratio_to_sample * len(pcd.points)):
            i_end = int(ratio_to_sample * len(pcd.points))

        pcd_batch = torch.Tensor(patches[i_start:i_end]).cuda()
        with torch.no_grad():
            f, _, _ = net(pcd_batch)

        pcd_desc[i_start:i_end] = f.cpu().detach().numpy()[:i_end - i_start]

    full_pcd_desc = np.empty((xyz.shape[0], dim))
    full_pcd_desc[voxel_idx] = pcd_desc

    tree = o3d.geometry.KDTreeFlann(pcd)

    for f in range(xyz.shape[0]):
        if f not in voxel_idx:
            _, knn_idx, _ = tree.search_knn_vector_3d(query=full_pcd.points[f], knn=1)
            full_pcd_desc[f] = pcd_desc[knn_idx]

    np.savez(fdip, features=full_pcd_desc.astype(float))


def main(args):
    seq = '08'
    root_seq_dir = os.path.join(args.source_path, seq, 'velodyne')
    lfpcds = glob.glob(os.path.join(root_seq_dir, '*.bin'))
    lfpcds.sort()

    dest_dir = os.path.join(args.save_path, seq)
    os.makedirs(dest_dir, exist_ok=True)

    splits = np.floor(np.linspace(0, len(lfpcds), 4)).astype(int)
    lfpcds = lfpcds[splits[args.split]:splits[args.split+1]]

    for nf, fpcd in enumerate(tqdm(lfpcds, desc='DIP')):
        pcd_id = os.path.basename(fpcd).split('.')[0]
        fdip = os.path.join(dest_dir, pcd_id + '.npz')

        if os.path.isfile(fdip):
            print(f'--> Skipping existing file {fdip}')
        else:
            t_exec = time.time()
            compute(fpcd, fdip)
            print('--> Done in {:.2f}s'.format(time.time() - t_exec))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

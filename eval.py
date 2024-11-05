'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''
#!/usr/bin/env python3
import os
import glob
import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
import trimesh
import pymeshlab
from models.archs.sdf_decoder import SdfDecoder
from models.archs.encoders.conv_pointnet import ConvPointnet, ConvPointnetDenseCls
from models.temporal_encoder import TemporalEncoder
from torch_scatter import scatter_sum
from kornia.geometry.conversions import quaternion_to_rotation_matrix, QuaternionCoeffOrder
from evaluation_metrics_3d import compute_all_metrics_4d
wxyz = QuaternionCoeffOrder.WXYZ


def standardize(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    standardized_data = (data - mean) / (std + 1e-8)
    return standardized_data

test_dir = "eval/test"

test_path = os.path.join(test_dir, "*.npy")
test_files = glob.glob(test_path)

refs = []
samples = []
for test_file in test_files:
    ref = np.load(test_file)
    refs.append(standardize(ref))
refs = np.array(refs)

latent_dim = 256
hidden_dim = 512
pn_hidden_dim = 128
num_layers = 9
skip_connection = True
tanh_act = False
pointnet = ConvPointnet(c_dim=latent_dim, hidden_dim=pn_hidden_dim, plane_resolution=64).cuda()
sdf_decoder = SdfDecoder(latent_size=latent_dim, hidden_dim=hidden_dim, skip_connection=skip_connection, tanh_act=tanh_act).cuda()

latent_dim = 64
k = 40
skinning_predictor = ConvPointnetDenseCls(k=40, c_dim=latent_dim, hidden_dim=pn_hidden_dim, plane_resolution=64).cuda()
shape_encoder = ConvPointnet(c_dim=latent_dim, hidden_dim=pn_hidden_dim, plane_resolution=64).cuda()
temporal_encoder = TemporalEncoder(input_size=latent_dim, add_linear=True, hidden_size=512, out_size=latent_dim).cuda()
transform_net = SdfDecoder(latent_size=latent_dim, hidden_dim=hidden_dim, skip_connection=skip_connection, tanh_act=tanh_act, input_size=1 + latent_dim, output_size=7).cuda()

pointnet.load_state_dict(torch.load("results/weights/pointnet.pth")) 
sdf_decoder.load_state_dict(torch.load("results/weights/sdf_decoder.pth"))
skinning_predictor.load_state_dict(torch.load("results/weights/skinning_predictor.pth"))
shape_encoder.load_state_dict(torch.load("results/weights/shape_encoder.pth"))
temporal_encoder.load_state_dict(torch.load("results/weights/temporal_encoder.pth")) 
transform_net.load_state_dict(torch.load("results/weights/transform_net.pth"))
plane_features = torch.load('results/features/plane_features_uncond.pth')
motion_features = torch.load('results/features/motion_features_uncond.pth')

def handle2mesh(transformation, handle_pos, region_score, batch, v0):
    """
    use per-part trans+rot to reconstruct the mesh
    transformation: (B, 40, 3+4)
    handle_pos: handle position of T-pose
    handle_pos: (B, 40, 3)
    region_score: (B*V, 40)
    v0: (B*V, 3)
    """
    B, K, _ = handle_pos.shape
    disp = transformation[:, :, :3]
    rot = transformation[:, :, 3:]
    rot = quaternion_to_rotation_matrix(rot.view(B*K, 4).contiguous(), order=wxyz).view(B, K, 3, 3).contiguous()
    hd_disp = torch.repeat_interleave(disp, torch.bincount(batch), dim=0)  # (B*V, 40, 3)
    hd_rot = torch.repeat_interleave(rot, torch.bincount(batch), dim=0)  # (B*V, 40, 3, 3)
    hd_pos = torch.repeat_interleave(handle_pos, torch.bincount(batch), dim=0)  # (B*V, 40, 3)
    per_hd_v = torch.einsum("abcd,abd->abc", hd_rot, (v0[:, None] - hd_pos)) + hd_pos + hd_disp  # (B*V, 40, 3)
    v = torch.sum(region_score[:, :, None] * per_hd_v, 1)  # (B*V, 3)
    return v

def create_mesh(
    decoder, filename, frame, idx, N=256, max_batch=64 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    voxel_origin = [-0.5, -0.5, -0.5]
    voxel_size = 1.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        shape_features = pointnet.forward_with_plane_features(plane_features[idx:idx+1].cuda(), sample_subset.unsqueeze(dim=0))
        samples[head : min(head + max_batch, num_samples), 3] = (
            decoder(torch.cat((sample_subset.unsqueeze(dim=0), shape_features), dim=-1))
            .squeeze()
            .detach()
            .cpu()
        )

        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_size,
        frame,
        idx,
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_size,
    frame,
    idx,
):

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    batch = torch.zeros(len(verts)).long().cuda()
    new_verts = (torch.from_numpy(verts).float() - 0.5).unsqueeze(dim=0).cuda()
    skinning_weights = skinning_predictor(new_verts).view(-1, k)
    score = skinning_weights / torch.repeat_interleave(scatter_sum(skinning_weights, batch, dim=0), torch.bincount(batch), dim=0)
    weighted_pos = score[:,:,None] * new_verts.reshape(-1,3)[:, None]
    weighted_pos = scatter_sum(weighted_pos, batch, dim=0)
    motion_feature = motion_features[idx:idx+1].reshape(1,40,64).cuda()
    all_verts = []
    with torch.no_grad():
        for frame in range(16):
            t = torch.zeros_like(motion_feature[:,:,0:1]) + frame/16
            t = t.cuda()
            trans = transform_net(torch.cat((t, motion_feature), dim=-1))
            pred = handle2mesh(trans, weighted_pos, skinning_weights, batch, new_verts.reshape(-1, 3))
            verts = (pred.double() + 0.5).cpu().detach().numpy()
            all_verts.append(verts)

    all_verts_smooth = []
    for verts in all_verts:
        mesh_set = pymeshlab.MeshSet()
        mesh = pymeshlab.Mesh(verts, faces)
        mesh_set.add_mesh(mesh)
        # for i in range(10):
        #     mesh_set.apply_filter('apply_coord_hc_laplacian_smoothing')
        mesh = mesh_set.current_mesh()
        verts = mesh.vertex_matrix()
        faces = mesh.face_matrix()
        all_verts_smooth.append(verts)

    v_min, v_max = float("inf"), float("-inf")
    pc_per_anim = []
    for verts in all_verts_smooth:
        vert_data_copy = verts
        vert = vert_data_copy - np.mean(vert_data_copy, axis=0, keepdims=True)
        v_min = min(v_min, np.amin(vert))
        v_max = max(v_max, np.amax(vert))
    for verts in all_verts_smooth:
        obj = trimesh.Trimesh(verts, faces)
        pc  = obj.sample(2048)
        pc_per_anim.append(pc)
    samples.append(standardize(np.array(pc_per_anim)))

for idx in range(265):
    os.makedirs('results/mesh/{}'.format(idx), exist_ok=True)
    create_mesh(sdf_decoder, 'results/mesh/{}/mesh_{}_{}'.format(idx, idx, 0), 0, idx)

metrics = compute_all_metrics_4d(torch.tensor(refs).float().cuda(), torch.tensor(samples).float().cuda(), 16)
print(metrics)
import argparse
import json
import numpy as np
import torch
import torchvision
import os
from pathlib import Path
import PIL.Image as Image
import torchvision.transforms as T
from tqdm import tqdm
import yaml



from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras

parser = argparse.ArgumentParser(description='Load depth images from a trained NeRF')
parser.add_argument('--config-path', type=str, required=True)
parser.add_argument('--savedir', type=str, required=True)
parser.add_argument('--transform', type=str, required=True)
args = parser.parse_args()

_, pipeline, _, _ = eval_setup(Path(args.config_path))
pipeline.model.eval()
train_cameras = pipeline.datamanager.train_dataset.cameras
train_cameras_optimizer = pipeline.datamanager.train_camera_optimizer
dp_output = pipeline.datamanager.train_dataparser_outputs

train_paths = pipeline.datamanager.train_dataset._dataparser_outputs.image_filenames
train_names = [train_paths[i].name for i in range(len(train_paths))]
eval_cameras = pipeline.datamanager.eval_dataset.cameras
eval_cameras_optimizer = pipeline.datamanager.eval_camera_optimizer
eval_paths = pipeline.datamanager.eval_dataset._dataparser_outputs.image_filenames
eval_names = [eval_paths[i].name for i in range(len(eval_paths))]

os.makedirs(args.savedir + '/rgb', exist_ok=True)
os.makedirs(args.savedir + '/depth', exist_ok=True)
os.makedirs(args.savedir + '/rgb_out', exist_ok=True)

data_paths = train_paths + eval_paths

transform_file = open(args.transform)
data = json.load(transform_file)
K = torch.tensor([[data['fl_x'], 0, data['cx']],
                  [0, data['fl_y'], data['cy']],
                  [0, 0, 1]])

f = open(args.savedir + "/utensils.gt.sim", 'w')
transforms = {}

def visercam_to_ns(c2w):
    dp_outputs = pipeline.datamanager.train_dataparser_outputs
    foobar = np.concatenate([dp_outputs.dataparser_transform.numpy(), np.array([[0, 0, 0, 1]])], axis=0)
    c2w = foobar @ np.concatenate([c2w, np.array([[0, 0, 0, 1]])], axis=0)
    c2w = c2w[:3]
    return c2w


# data = data['frames']

# for entry in tqdm(data):
#     transforms[entry['file_path']] = entry['transform_matrix']

# for path in data_paths:
#     path = path.name
#     m = np.array(transforms[path])
#     m = m[:3]
#     # m = visercam_to_ns(m)
#     f.write(str(m[0])[1:-1])
#     f.write('\n')
#     f.write(str(m[1])[1:-1])
#     f.write('\n')
#     f.write(str(m[2])[1:-1])
#     f.write('\n')
#     f.write('\n')

# f.close()

yml_path = open(args.savedir + "/icl.yaml", 'w')
config = {'dataset_name': 'icl',
          'camera_params': {
              'image_height': data['h'],
              'image_width' : data['w'],
              'fx': data['fl_x'],
                'fy': data['fl_y'],
                'cx': data['cx'],
                'cy': data['cy'],
                'png_depth_scale': 1,
                'crop_edge': 0
          }
          }
yaml.dump(config, yml_path)

for i, path in tqdm(enumerate(train_paths + eval_paths)):
    path = str(path.absolute())
    img = torchvision.io.read_image(path)
    torchvision.io.write_png(img, args.savedir + '/rgb/' + f"{i}.png")

for i in tqdm(range(len(train_cameras))):
    with torch.no_grad():
        currcam = train_cameras[i]
        cam_opt = train_cameras_optimizer([i]).squeeze() #Transformation matrices from optimized camera coordinates to given camera coordinates (3, 4).
        transformed_cam = torch.concat([currcam.camera_to_worlds.cuda(), torch.tensor([[0, 0, 0, 1]]).cuda()], axis=0) @ torch.concat([cam_opt, torch.tensor([[0, 0, 0, 1]]).cuda()], axis=0)
        currcam.camera_to_worlds = transformed_cam[:3].cpu()
        bundle = currcam.generate_rays(camera_indices=0)
        bundle = bundle.to(pipeline.device)
        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(bundle)

    m = np.array(currcam.camera_to_worlds)
    f.write(str(m[0])[1:-1])
    f.write('\n')
    f.write(str(m[1])[1:-1])
    f.write('\n')
    f.write(str(m[2])[1:-1])
    f.write('\n')
    f.write('\n')

    img = outputs["rgb"].permute(2, 0 ,1).cpu()
    img = img*255 
    img = img.to(torch.uint8)
    torchvision.io.write_png(img, args.savedir + '/rgb_out/' + f"{i}.png")

    distance = outputs["depth_med"]
    h, w, _ = distance.shape
    coords = np.ones((h, w, 3))
    coords[:, :, :2] = np.mgrid[:w, :h].T # (h, w, 2)

    # Apply back-projection: multiply the inverse of the camera matrix
    # by the pixel coordinates to obtain the back-projected rays in
    # world space.
    rays_d = np.einsum("ij,hwj->hwi", np.linalg.inv(K), coords)
    rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True) # (h, w, 3)
    depths = rays_d[..., 2] * np.array(distance.squeeze().cpu())

    # img = (outputs["depth"] / dp_output.dataparser_scale).squeeze().cpu()
    # distance = distance.squeeze().cpu().numpy()
    np.save(args.savedir + '/depth/' + f"{i}", depths)

    # img = (outputs["depth"] / dp_output.dataparser_scale).squeeze().cpu()
    # img = outputs["depth"].permute(2, 0 ,1).cpu()
    # x = T.ToPILImage()(img)
    # x.save(args.savedir + '/depth/' + f"{i}.png")



for i in tqdm(range(len(eval_cameras))):
    with torch.no_grad():
        currcam = eval_cameras[i]
        cam_opt = eval_cameras_optimizer([i]).squeeze() #Transformation matrices from optimized camera coordinates to given camera coordinates (3, 4).
        transformed_cam = torch.concat([currcam.camera_to_worlds.cuda(), torch.tensor([[0, 0, 0, 1]]).cuda()], axis=0) @ torch.concat([cam_opt, torch.tensor([[0, 0, 0, 1]]).cuda()], axis=0)
        currcam.camera_to_worlds = transformed_cam[:3].cpu()
        bundle = currcam.generate_rays(camera_indices=0)
        bundle = bundle.to(pipeline.device)
        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(bundle)
    
    m = np.array(currcam.camera_to_worlds)
    f.write(str(m[0])[1:-1])
    f.write('\n')
    f.write(str(m[1])[1:-1])
    f.write('\n')
    f.write(str(m[2])[1:-1])
    f.write('\n')
    f.write('\n')

    img = outputs["rgb"].permute(2, 0, 1).cpu()
    img = img*255 
    img = img.to(torch.uint8)
    torchvision.io.write_png(img, args.savedir + '/rgb_out/' + f"{i+len(train_cameras)}.png")

    distance = outputs["depth_med"]
    h, w, _ = distance.shape
    coords = np.ones((h, w, 3))
    coords[:, :, :2] = np.mgrid[:w, :h].T # (h, w, 2)

    # Apply back-projection: multiply the inverse of the camera matrix
    # by the pixel coordinates to obtain the back-projected rays in
    # world space.
    rays_d = np.einsum("ij,hwj->hwi", np.linalg.inv(K), coords)
    rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True) # (h, w, 3)
    depths = rays_d[..., 2] * np.array(distance.squeeze().cpu())

    # img = (outputs["depth"] / dp_output.dataparser_scale).squeeze().cpu()
    # distance = distance.squeeze().cpu().numpy()
    np.save(args.savedir + '/depth/' + f"{i+len(train_cameras)}", depths)

    # img = outputs["depth"].permute(2, 0 ,1).cpu()
    # x = T.ToPILImage()(img)
    # x.save(args.savedir + '/depth/' + f"{i+len(train_cameras)}.png")

f.close()
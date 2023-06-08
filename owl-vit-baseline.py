
import requests
from PIL import Image
import torch
import cv2
import numpy as np
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import argparse
import json
import numpy as np

def project_to_image(self, K, point_cloud, round_px=True):
    points_proj = torch.matmul(K, point_cloud)
    if len(points_proj.shape) == 1:
        points_proj = points_proj[:, None]
    point_depths = points_proj[2, :]
    point_z = point_depths.repeat(3, 1)
    points_proj = torch.divide(points_proj, point_z)
    if round_px:
        points_proj = torch.round(points_proj)
    points_proj = points_proj[:2, :].int()

    # valid_ind = torch.where((points_proj[0, :] >= 0) & (points_proj[1, :] >= 0) & (points_proj[0, :] < curcam.image_width.item()) & (points_proj[1, :] < curcam.image_height.item()), 1, 0).to(device)
    # valid_ind = torch.argwhere(valid_ind.squeeze())
    # depth_data = torch.zeros([curcam.image_height, curcam.image_width]).to(device)
    
    # depth_data[points_proj[1, valid_ind], points_proj[0, valid_ind]] = point_depths[valid_ind]
    return points_proj


def check_if_pt_in_bbox(pt, bbox):
    if pt[0] > bbox[0] and pt[0] < bbox[2] and pt[1] > bbox[1] and pt[1] < bbox[3]:
        return True
    return False

def grasps_in_image(image, grasps_world, cam2world, K_matrix):
    homog_cam_to_world = torch.cat((mod_curcam.camera_to_worlds.squeeze(), torch.tensor([[0,0,0,1]]).to(device)), dim=0)
    homog_world_to_cam = torch.inverse(homog_cam_to_world)
    pcd_points = np.asarray(pcd.points)
    homog_pcd_points = np.ones((pcd_points.shape[0],4))
    homog_pcd_points[:, :3] = pcd_points
    # x_axis = RigidTransform.x_axis_rotation(np.pi)
    # rotation_matrix_x = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], device=device).float() #180 deg rot about cam x axis
    # rotation_matrix_y = torch.tensor([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], device=device).float() #180 deg rot about cam x axis
    # reflection_matrix_xy = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], device=device).float() #180 deg rot about cam x axis
    point_cloud_in_camera_frame = torch.matmul(homog_world_to_cam, torch.tensor(homog_pcd_points.T, device=device).float())#Don't x axis rotation is necessary here
    all_grasps_pixel = project_to_image(K_matrix, point_cloud_in_camera_frame[:3, :])

    #Visualization----
    valid_ind = torch.where((points_proj[0, :] >= 0) & (points_proj[1, :] >= 0) & (points_proj[0, :] < curcam.image_width.item()) & (points_proj[1, :] < curcam.image_height.item()), 1, 0).to(device)
    valid_ind = torch.argwhere(valid_ind.squeeze())
    depth_data = torch.zeros([curcam.image_height, curcam.image_width]).to(device)
    depth_data[all_grasps_pixel[1, valid_ind], all_grasps_pixel[0, valid_ind]] = point_depths[valid_ind]
    grasp_masks = torch.where(depth_data > 0, 1, 0) 
    image_point = grasp_masks.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float()
    kernel = torch.ones(3, 3).to(device)
    grasp_masks = kmorph.dilation(image_point, kernel)
    torchvision.utils.save_image(grasp_masks.squeeze(), "./grasp.png")
    return all_grasps_pixel

def load_poses(pose_path):
    poses = []

    lines = []
    with open(pose_path, "r") as f:
        lines = f.readlines()

    _posearr = []
    for line in lines:
        line = line.strip().split()
        if len(line) == 0:
            continue
        _npvec = np.asarray(
            [float(line[0]), float(line[1]), float(line[2]), float(line[3])]
        )
        _posearr.append(_npvec)
    _posearr = np.stack(_posearr)

    for pose_line_idx in range(0, _posearr.shape[0], 3):
        _curpose = np.zeros((4, 4))
        _curpose[3, 3] = 3
        _curpose[0] = _posearr[pose_line_idx]
        _curpose[1] = _posearr[pose_line_idx + 1]
        _curpose[2] = _posearr[pose_line_idx + 2]
        poses.append(torch.from_numpy(_curpose).float())
    return poses

parser = argparse.ArgumentParser(description='Transforms a set of poses from one frame to another')
# parser.add_argument('--output_folder', type=str, required=True)
parser.add_argument('--depth_folder', type=str, required=True)
parser.add_argument('--save_folder', type=str, required=True)
K_matrix = torch.load(args.depth_folder + '/k.pt')
data = json.load(transform_file)
data = data['frames']
image_poses = []
for i, frame in enumerate(data):
    m = np.array(frame['transform_matrix'])
    m = m[:3]
    image = Image.open(args.depth_folder + '/' + 'img' + str(i) + '.png')
    image_poses.append((image, m))
    
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
image = Image.open("./img9.png")
texts = [["dish scrub brush", "dish scrub brush handle"]]#,"dust pan brush", "dust pan brush handle", "shiny black spoon handle", 
        #   "matte black spoon handle", "teapot", "matte black spoon handle"]]

from graspnetAPI import GraspGroup
grasps = GraspGroup().from_numpy("./scene_1_grasps.npy")

grasps_world = [] #world grasps in xyz coords
best_grasps_for_each_label = {}
for image, pose in image_poses:
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)
    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    image2 = np.asarray(image)
    best_for_each_label = {}
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if text[label] not in best_for_each_label.keys() or score > best_for_each_label[text[label]][1]:
            center = ((box[0] + box[2])/2, (box[1] + box[3])/2)
            best_for_each_label[text[label]] = (box, score, center)
    print(best_for_each_label)
    grasps_pixel =  grasps_in_image(image, grasps_world, pose) #dimenions np.arrays world graps X 1
    for label, (box, score, center) in best_for_each_label.items():
        print(f"Detected {label} with confidence {round(score.item(), 3)} at location {box}")
        cv2.rectangle(image2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        cv2.putText(image2, label, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        valid_ind = np.where(grasps_pixel[:, 0] < box[2] & grasps_pixel[:, 0] > box[0] & grasps_pixel[:, 1] < box[3] & grasps_pixel[:, 1] > box[1])
        if label not in best_grasps_for_each_label.keys():
            best_grasps_for_each_label[label] = np.zeros(len(grasps_pixel))
        best_grasps_for_each_label[label][valid_ind] += 1
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    cv2.imwrite(args.save_folder + '/' + 'img' + str(i) + '.png',image2)      


best_grasps_for_each_label_final = {k:grasps_world[np.argmax(v)] for k,v in best_grasps_for_each_label.items()}
print(best_grasps_for_each_label_final)
#then center of bbox is target point
#get the depth from the image project tp in pc
#get distance away from grasp
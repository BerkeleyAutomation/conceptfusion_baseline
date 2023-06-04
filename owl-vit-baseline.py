
import requests
from PIL import Image
import torch
import cv2
import numpy as np
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import argparse
import json
import numpy as np

def check_if_pt_in_bbox(pt, bbox):
    if pt[0] > bbox[0] and pt[0] < bbox[2] and pt[1] > bbox[1] and pt[1] < bbox[3]:
        return True
    return False

def grasps_in_image(image, grasps_world, pose):
    return grasps_pixel

parser = argparse.ArgumentParser(description='Transforms a set of poses from one frame to another')
parser.add_argument('--output_folder', type=str, required=True)
parser.add_argument('--save_folder', type=str, required=True)
args = parser.parse_args()
transform_file = open(args.output_folder + '/transforms.json')
data = json.load(transform_file)
data = data['frames']
image_poses = []
for i, frame in enumerate(data):
    m = np.array(frame['transform_matrix'])
    m = m[:3]
    image = Image.open(args.output_folder + '/' + 'img' + str(i) + '.png')
    image_poses.append((image, m))
    
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
image = Image.open("./img9.png")
texts = [["dish scrub brush", "dish scrub brush handle"]]#,"dust pan brush", "dust pan brush handle", "shiny black spoon handle", 
        #   "matte black spoon handle", "teapot", "matte black spoon handle"]]
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
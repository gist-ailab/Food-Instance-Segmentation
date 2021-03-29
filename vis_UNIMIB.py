import os
import cv2
import json
import numpy as np

from skimage.draw import polygon2mask
import skimage

def cvt_str(x):
    return str(x[0][0])

# LOAD MAT fiel and convert into List[Str]
print("---" * 15)
from scipy.io import loadmat
data_root = "/data/joo/dataset/food/UNIMIB2016/split"
ann_file = os.path.join(data_root, 'TestSet.mat')
anns = loadmat(ann_file)['TestSet']
test_set = list(map(cvt_str, anns))
print("TestSet", len(test_set))

ann_file = os.path.join(data_root, 'TrainingSet.mat')
anns = loadmat(ann_file)['TrainingSet']
train_set = list(map(cvt_str, anns))
print("TrainingSet", len(train_set))

exit()

# read txt file and save as JSON
def remove_bracket(ann):
    return ann.replace('[', '')
def cvt_list(ann):
    if ann[0] == ',':
        ann = ann[1:]
    return list(map(int, ann.split(',')))
save_root = "/data/joo/dataset/food/UNIMIB2016/annotations"
txt_root = "/data/joo/dataset/food/UNIMIB2016/annotations_txt"
# load trainset
poly_dict = {}
for img_name in train_set:
    print("IMG: ", img_name)
    txt_name = os.path.join(txt_root, "{}.txt".format(img_name))
    ann = open(txt_name, 'r').readline()
    # print(ann)
    ann = ann.split(']')[:-2]
    ann = list(map(remove_bracket, ann))
    ann = list(map(cvt_list, ann))
    print("Instance: ", len(ann))
    for inst_ann in ann:
        print(len(inst_ann), inst_ann[:10], "...")
    print("-----" * 25)
    poly_dict[img_name] = ann
# save as json file
save_name = os.path.join(save_root, "train.json")
with open(save_name, "w") as json_file:
    json.dump(poly_dict, json_file)

# load testset
poly_dict = {}
for img_name in test_set:
    print("IMG: ", img_name)
    txt_name = os.path.join(txt_root, "{}.txt".format(img_name))
    ann = open(txt_name, 'r').readline()
    # print(ann)
    ann = ann.split(']')[:-2]
    ann = list(map(remove_bracket, ann))
    ann = list(map(cvt_list, ann))
    print("Instance: ", len(ann))
    for inst_ann in ann:
        print(len(inst_ann), inst_ann[:10], "...")
    print("-----" * 25)
    poly_dict[img_name] = ann
# save as json file
save_name = os.path.join(save_root, "test.json")
with open(save_name, "w") as json_file:
    json.dump(poly_dict, json_file)
exit()








# data_root = "/data/joo/dataset/food/UNIMIB2016"
# ann_file = os.path.join(data_root, "annotations/val.json")

# anns = json.load(open(ann_file))
# print(type(anns))

# # print(anns.keys())
# # print(len(anns))


# data_root = "/data/joo/dataset/food/UNIMIB2016"
# modes = ['train', 'val', 'test']
# for mode in modes:
#     ann_file = os.path.join(data_root, "annotations/{}.json".format(mode))
#     anns = json.load(open(ann_file))
#     print(mode, len(anns))

#     for img_name, ann in anns.items():
#         num_instance = len(ann)
#         img_file = os.path.join(data_root, "images", "{}.jpg".format(img_name))
#         img_arr = cv2.imread(img_file)
#         img_h, img_w, img_c = img_arr.shape
#         cv2.imwrite("tmp_IMG.png", img_arr)
#         # mask_arr = np.zeros((img_h, img_w, num_instance), dtype=np.uint8)
#         mask_arr = np.zeros((img_h, img_w), dtype=np.uint8)
#         print("IMG: ", img_name, img_arr.shape)
#         for ann_inst in ann:
#             poly_dict = list(ann_inst.values())[0]
#             poly_pts = poly_dict["BR"]
#             rr, cc = skimage.draw.polygon(poly_pts[1::2], poly_pts[::2])
#             rr = (img_h-1) - rr
#             cc = (img_w-1) - cc
#             mask_arr[rr, cc] = 150
#             cv2.imwrite("tmp_{}.png".format(food), mask_arr)
#         exit()

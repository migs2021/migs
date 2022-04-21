
import os
import json
import argparse
from tqdm import tqdm
import collections
from random import shuffle

parser = argparse.ArgumentParser(description='bdd2coco')
parser.add_argument('--bdd_dir', type=str, default='./bdd100k_dataset/')
parser.add_argument('--base_dir', type=str, default="./")
cfg = parser.parse_args()

src_val_dir = os.path.join(cfg.bdd_dir, 'labels', 'bdd100k_labels_images_val.json')
src_train_dir = os.path.join(cfg.bdd_dir, 'labels', 'bdd100k_labels_images_train.json')

os.makedirs(os.path.join("./", 'labels_coco'), exist_ok=True)

dst_val_dir = os.path.join("./", 'bdd_test', 'bdd100k_labels_images_val_coco.json')
dst_train_dir = os.path.join("./", 'bdd_test', 'bdd100k_labels_images_train_coco.json')

dst_meta_train =  os.path.join("./", 'bdd_test', 'bdd100k_meta_train.json')
dst_meta_val =  os.path.join("./", 'bdd_test', 'bdd100k_meta_val.json')


def create_attr_dicts(data):
  timeofday = ['daytime', 'night', 'dawn/dusk', 'undefined']

  datanums_train = {}
  datanums_test = {}
  for name in timeofday:
    #     for scene in scenes:
    datanums_train[name] = collections.defaultdict(dict)
    datanums_test[name] = collections.defaultdict(dict)

  datanums_val = {}
  for name in timeofday:
    #     for scene in scenes:
    datanums_val[name] = collections.defaultdict(dict)

  val_entr =[]
  train_entr = []
  for entr in data:
    attrs = entr['attributes']
    if entr['name'] == '23b74d79-dfeb6075.jpg':
      continue
    if attrs['timeofday'] in datanums_val.keys() and attrs['weather'] in datanums_val[attrs['timeofday']].keys() and attrs[
      'scene'] in datanums_val[attrs['timeofday']][attrs['weather']].keys():
      datanums_val[attrs['timeofday']][attrs['weather']][attrs['scene']].append(entr)
      val_entr.append(entr)
    else:
      datanums_val[attrs['timeofday']][attrs['weather']][attrs['scene']] = []

  for weather in datanums_val.keys():
    for scene in datanums_val[weather].keys():
      for timeofday in datanums_val[weather][scene].keys():
          len_sub = len(datanums_val[weather][scene][timeofday])
          print(len_sub)
          datanums_train[weather][scene][timeofday] = datanums_val[weather][scene][timeofday][:min(160, len_sub)]
          datanums_test[weather][scene][timeofday] = datanums_val[weather][scene][timeofday][min(160, len_sub):]

  return datanums_train, train_entr, datanums_test, val_entr

def bdd2coco_detection(datanums, suff):
  attr_dict = {"categories":
    [
      {"supercategory": "none", "id": 1, "name": "person"},
      {"supercategory": "none", "id": 2, "name": "car"},
      {"supercategory": "none", "id": 3, "name": "rider"},
      {"supercategory": "none", "id": 4, "name": "bus"},
      {"supercategory": "none", "id": 5, "name": "truck"},
      {"supercategory": "none", "id": 6, "name": "bike"},
      {"supercategory": "none", "id": 7, "name": "motor"},
      {"supercategory": "none", "id": 8, "name": "traffic light"},
      {"supercategory": "none", "id": 9, "name": "traffic sign"},
      # {"supercategory": "none", "id": 10, "name": "train"},
    ]}

  id_dict = {i['name']: i['id'] for i in attr_dict['categories']}

  images = []
  annotations = []
  ignore_categories = set()

  counter = 0
  for weather in datanums.keys():
    for scene in datanums[weather].keys():
      for timeofday in datanums[weather][scene].keys():
        if (suff == 'train' and len(datanums[weather][scene][timeofday]) == 160) or (suff == 'test' and len(datanums[weather][scene][timeofday]) > 100):
          file_to_save = os.path.join(cfg.base_dir, 'bdd_full_dset_test', suff, "{}_{}_{}.json".format(weather.replace('/',''),scene.replace(' ',''), timeofday))
          local_images = list()
          local_annotations = list()
          local_categories = set()
          local_counter = 0
          for i in datanums[weather][scene][timeofday]:
            local_counter += 1
            image = dict()
            image['file_name'] = i['name']
            image['height'] = 720
            image['width'] = 1280

            image['id'] = counter

            empty_image = True

            tmp = 0
            for l in i['labels']:
              annotation = dict()
              if l['category'] in id_dict.keys():
                tmp = 1
                empty_image = False
                annotation["iscrowd"] = 0
                annotation["image_id"] = image['id']
                x1 = l['box2d']['x1']
                y1 = l['box2d']['y1']
                x2 = l['box2d']['x2']
                y2 = l['box2d']['y2']
                annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]
                annotation['area'] = float((x2 - x1) * (y2 - y1))
                annotation['category_id'] = id_dict[l['category']]
                annotation['ignore'] = 0
                annotation['id'] = l['id']
                annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                # annotations.append(annotation)


                annotation["image_id"] = local_counter
                local_annotations.append(annotation)
              else:
                ignore_categories.add(l['category'])
                local_categories.add(l['category'])

            if empty_image:
              print('empty image!')
              continue
            if tmp == 1:
              # images.append(image)
              image['id'] = local_counter
              local_images.append(image)

          attr_dict["images"] = local_images
          attr_dict["annotations"] = local_annotations
          attr_dict["type"] = "instances"

          print('ignored categories: ', ignore_categories)
          print('saving...{} {} {}'.format(weather,scene, timeofday))
          with open(file_to_save, "w") as file:
            json.dump(attr_dict, file)
 

def main():
  # create BDD training set detections in COCO format
  print('Loading training set...')
  with open(src_train_dir) as f:
    train_labels = json.load(f)
  print('Converting training set...')
  train_datanums, train_entr, val_datanums, val_entr = create_attr_dicts(train_labels)
  bdd2coco_detection(train_datanums, 'train')
  bdd2coco_detection(val_datanums, 'test')


if __name__ == '__main__':
  main()
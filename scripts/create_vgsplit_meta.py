import json, os

VG_DIR = '/media/azadef/MyHDD/Code/GANs/SIMSG/sg2im/datasets/vg'

clusters = os.path.join(VG_DIR, 'pred_clusters_vg.json')
images_json = os.path.join(VG_DIR, 'image_data.json')
new_split_file = os.path.join(VG_DIR, 'vg_meta_splits.json')

with open(clusters, 'r') as f:
    cluster_assignments = json.load(f)

with open(images_json, 'r') as f:
    images = json.load(f)


image_id_to_image = {i['url'].split("/")[-1]: i for i in images}

split_dict = {}
for im_name, cluster_id in cluster_assignments.items():
    if str(cluster_id) not in split_dict.keys():
        split_dict[str(cluster_id)] = []

    im_id = int(im_name.split(".")[0])
    if im_name in image_id_to_image.keys():
        split_dict[str(cluster_id)].append(im_id)

with open(new_split_file, 'w') as f:
    json.dump(split_dict, f)

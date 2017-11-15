import csv
import sys
import glob
import yaml
from itertools import chain
import numpy as np
from libdvid import DVIDNodeService

dirpath = '/magnetic/workspace/DVIDSparkServices/integration_tests/test_copyseg_remapped'

configs = glob.glob(dirpath + "/temp_data/config.*")
assert len(configs) == 1, "Why does the temp_dir have more than one config.* file?"

with open(configs[0], 'r') as f:
    config = yaml.load(f)

input_service = DVIDNodeService(str(config['input']['source']['server']), str(config['input']['source']['uuid']))
input_name = config['input']['source']['segmentation-name']
input_bb_xyz = config['input']['geometry']['bounding-box']
input_bb_zyx = np.array(input_bb_xyz)[:,::-1]
input_shape = input_bb_zyx[1] - input_bb_zyx[0]

input_volume = input_service.get_labels3D(input_name, input_shape, input_bb_zyx[0])

def get_output_vol(index):
    output_service = DVIDNodeService(str(config['outputs'][index]['source']['server']), str(config['outputs'][index]['source']['uuid']))
    output_name = config['outputs'][index]['source']['segmentation-name']
    output_bb_xyz = config['outputs'][index]['geometry']['bounding-box']
    output_bb_zyx = np.array(output_bb_xyz)[:,::-1]
    output_shape = output_bb_zyx[1] - output_bb_zyx[0]
    
    output_volume = output_service.get_labels3D(output_name, output_shape, output_bb_zyx[0])
    return output_volume

output_vols = list(map(get_output_vol, range(3)))

# The mappings in this test amount to adding 100+200 (input_mapping + output_mapping) to every value
if not (input_volume + 300 == output_vols[0]).all():
    print("DEBUG: FAIL: output volume 0 does not correspond to remapped input volume!")
    sys.exit(1)

if not (input_volume + 500 == output_vols[1]).all():
    print("DEBUG: FAIL: output volume 1 does not correspond to remapped input volume!")
    sys.exit(1)

# The third output volume should be one giant merged body
if len(np.unique(output_vols[2])) > 1:
    print("DEBUG: FAIL: output volume 2 should be a giant merged body")
    sys.exit(1)

csv_path = dirpath + '/output_totalmerge_edges.csv'
with open(csv_path, 'r') as csv_file:
    rows = csv.reader(csv_file)
    all_items = chain.from_iterable(rows)
    edges = np.fromiter(all_items, np.uint64).reshape(-1,2) # implicit conversion from str -> uint64

min_edge_id = edges.min()
if not (min_edge_id == output_vols[2]).all():
    print("DEBUG: FAIL: output volume 2 should be a giant merged body whose ID is the minimum ID found in the mapping file.")
    sys.exit(1)

print("DEBUG: CopySegmentation (with remapping) test passed.")
sys.exit(0)

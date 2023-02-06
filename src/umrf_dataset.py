import os
import glob2 as glob

import json
import csv

from torch.utils.data import Dataset, DataLoader

"""
This class is used to manage access to the supervised UMRF examples. These examples were
developed by taking the natural language (NL) instructions used in ALFRED and developing
UMRF parses.
"""


class UMRF(Dataset):

    def __init__(self, data_path: str):
        self.umrf_data_path = data_path
        self.all_umrf_examples_path = sorted(glob.glob(data_path))

    def __len__(self) -> int:
        return len(self.all_umrf_examples_path)

    """
    Provide an index for a UMRF example and get in return:
    
    + Natural Language Instruction (string)
    + Image Data (coordinate information as a string of Pose2D coords) & visual landmark info
    + UMRF Graph (the ground truth label/ decoding)
    """

    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        umrf_ex_path = self.all_umrf_examples_path[idx]
        umrf = self.path_to_umrf(umrf_ex_path)

        nl_instruction = str(umrf['graph_description'])
        del umrf['graph_description']
        image_data = self.get_image_data(umrf)

        return nl_instruction, image_data, str(umrf)

    def path_to_umrf(self, path: str) -> json:
        with open(path, 'r') as f:
            json_string = f.read()
            json_dict = json.loads(json_string)
            del json_dict['graph_name']
            del json_dict['graph_state']
        return json_dict

    def get_image_data(self, umrf: json) -> str:
        is_simplified = True
        umrf_actions = umrf['umrf_actions']
        coordinate_data = []
        for action in umrf_actions:
            try:
                # keeps format in the json style just in case using ROS-like messaging
                # for pose information
                if not is_simplified:
                    coordinate_data.append(
                        str(action['input_parameters']['pose_2d']))
                else:
                    coordinate_data.append(
                        self.simplify_viz(action['input_parameters']['pose_2d']))
            except:
                pass
            try:
                # no json formatting for list of relevant locations
                coordinate_data.append(
                    str(action['input_parameters']['location']['pvf_example']))
            except:
                pass
            try:
                # no json formatting for list of relevant landmarks
                coordinate_data.append(
                    str(action['input_parameters']['landmark']['pvf_example']))
            except:
                pass
        img_str = " ".join(coordinate_data)
        return img_str


    def simplify_viz(self, viz_info):
        # print(viz_info)
        # str_len = len(viz_info)
        # index = str_len
        # for char in reversed(viz_info):
        #     if char == '}':
        #         break
        #     index = index - 1
        # viz_json = json.loads(viz_info[0:index].replace("\'", "\""))

        x = viz_info['x']['pvf_value']
        y = viz_info['y']['pvf_value']
        yaw = viz_info['yaw']['pvf_value']

        simplified_coords = '[x={}; y={}; yaw={}]'.format(x, y, yaw)

        return simplified_coords

if __name__ == '__main__':
    umrf_data_path = os.getcwd() + '/umrf_data/*'
    print(umrf_data_path)
    dataset = UMRF(umrf_data_path)
    print(dataset[0])

    f = open('umrf.csv', 'w')
    header = ['prompt', 'target']
    writer = csv.writer(f)
    writer.writerow(header)
    for ex in dataset:
        nl_instruction = ex[0]
        visual_info = ex[1]
        label = ex[2]

        reformatted = [nl_instruction + ' ' + visual_info + ' ' + label]
        writer.writerow(reformatted)
    f.close()
import os

import pandas as pd

labels = dict(fp=[], xmin=[], xmax=[], ymin=[], ymax=[])

# Replace 'your_folder_path' with the path to the folder you want to open
folder_path = 'train/labels'

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Check if it's a file (not a subdirectory)
    if os.path.isfile(file_path):

        with open(file_path, 'r') as file:
            coordinates_string = file.read()

            # Split the string into a list of values and convert them to floats
            coordinates_list = list(map(float, coordinates_string.split()[1:]))
            labels['fp'].append(file_path)
            labels['xmin'].append(float(coordinates_list[0]))
            labels['xmax'].append(float(coordinates_list[1]))
            labels['ymin'].append(float(coordinates_list[2]))
            labels['ymax'].append(float(coordinates_list[3]))

df = pd.DataFrame(labels)
df.to_csv('data.csv',index=False)
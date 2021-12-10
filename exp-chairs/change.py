import json
from os.path import exists

train_split_file='./sv2_chairs_train.json'

with open(train_split_file, "r") as f:
    train_split = json.load(f)

    existing_files=[]
    for file in train_split['ShapeNetV2']['03001627']:
        if exists('../data/SdfSamples/ShapeNetV2/03001627/{}.npz'.format(file)):
            existing_files.append(file)

    train_split['ShapeNetV2']['03001627'] = existing_files
    
    print(len(train_split['ShapeNetV2']['03001627']))

    with open("new_split.json","w") as json_file:
        json.dump(train_split,json_file)

    import pdb; pdb.set_trace()

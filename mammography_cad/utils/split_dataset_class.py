import json
import random
import os


def split_dataset_class(json_path, out_dir, train_ratio=0.7, val_ratio=0.15):
    with open(json_path) as f:
        data = json.load(f)


    keys = list(data.keys())
    random.shuffle(keys)

    train_end = int(train_ratio * len(keys))
    val_end = train_end + int(val_ratio * len(keys))

    train_keys = keys[:train_end]
    val_keys = keys[train_end:val_end]
    test_keys = keys[val_end:]

    splits = {'train': train_keys, 'val': val_keys, 'test': test_keys}
    out_dir = "data/json/class"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for split, split_keys in splits.items():
        split_dict = {k: data[k] for k in split_keys}
        with open(f"{out_dir}/{split}.json", "w") as f:
            json.dump(split_dict, f, indent=4)
        print(f"{split}: {len(split_keys)} images")

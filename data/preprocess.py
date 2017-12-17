import numpy as np
import torch

import load_data
import realtime_augmentation as ra


# parameters related to data processing
input_sizes = [(69, 69), (69, 69)]
viewpoint_size = (45, 45)

ds_transforms = [
    ra.build_ds_transform(3.0, target_size=input_sizes[0]),
    ra.build_ds_transform(3.0, target_size=input_sizes[1]) + ra.build_augmentation_transform(rotation=45)
    ]

augmentation_params = {
    'zoom_range': (1.0 / 1.3, 1.3),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-4, 4),
    'do_flip': True,
}

GEN_BUFFER_SIZE = 1

# create 16 view point for each image
def create_viewpoints(data_gen, part_size = (45, 45)):
    for target_arrays, batch_size in data_gen:
        labels = target_arrays.pop()

        ps = part_size[0]
        parts = []
        for chunk_imgs in target_arrays:
            chunk_imgs = chunk_imgs.transpose((0, 3, 1, 2))
            chunk_imgs_flip = [chunk_imgs, chunk_imgs[:, :, :, ::-1]]
            for img_flip in chunk_imgs_flip:
                part0 = img_flip[:, :, :ps, :ps] # 0 degrees
                part1 = img_flip[:, :, :ps, :-ps-1:-1].transpose(0, 1, 3, 2) # 90 degrees
                part2 = img_flip[:, :, :-ps-1:-1, :-ps-1:-1] # 180 degrees
                part3 = img_flip[:, :, :-ps-1:-1, :ps].transpose(0, 1, 3, 2) # 270 degrees
                parts.extend([part0, part1, part2, part3])

        target_arrays = np.concatenate(parts, axis=1)

        yield (torch.from_numpy(target_arrays).float(), torch.from_numpy(labels).float())

# data loader, data_indices is the selected indeces from the training data set.
def dataLoader(data_indices, batch_size=100, shuffle=True):
    data_indices = np.array(data_indices)
    if shuffle:
        np.random.shuffle(data_indices)

    augmented_data_gen = ra.realtime_augmented_data_gen(data_indices = data_indices,
                                                        batch_size=batch_size,
                                                        augmentation_params=augmentation_params, ds_transforms=ds_transforms,
                                                        target_sizes=input_sizes)

    post_augmented_data_gen = ra.post_augment_brightness_gen(augmented_data_gen, std=0.5)

    viewpoints_gen = create_viewpoints(post_augmented_data_gen, part_size = viewpoint_size)

    data_gen = load_data.buffered_gen_mp(viewpoints_gen, buffer_size=GEN_BUFFER_SIZE)

    return data_gen

# example
if __name__ == '__main__':

    train_ids = load_data.train_ids
    num_train = len(train_ids)

    num_valid = num_train // 10 # integer division
    num_train -= num_valid

    train_indices = np.arange(num_train)
    valid_indices = np.arange(num_train, num_train + num_valid)

    for epoch in range(1, 2+1):
        print("\nepoch: %d" % epoch)

        # need to set train_loader and val_loader again at each epoch
        train_loader = dataLoader( data_indices = train_indices, batch_size=40, shuffle=True)
        val_loader = dataLoader( data_indices = valid_indices, batch_size=5, shuffle=False)

        print("training batches:")
        for batch_idx, (data, target) in enumerate(train_loader):
            print("batch: %d" % batch_idx)
            print(data.shape, target.shape)

        print("validation batches:")
        for batch_idx, (data, target) in enumerate(val_loader):
            print("batch: %d" % batch_idx)
            print(data.shape, target.shape)

from image_utils import save_exr, save_exr_one_channel
import os
import numpy as np
from image_utils import load_exr, load_exr_one_channel
import torch.utils.data as Data
import torch
import pickle


class DataHandler:
    def __init__(self,
                 data_dir,
                 subset,
                 scene_list,
                 patch_per_img=50,
                 patch_width=128,
                 patch_height=128):
        """
        - data_dir is location
        - subset use train|test
        - batch_size is int
        """
        self.data_dir = data_dir
        self.subset = subset
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.scene_list = scene_list
        self.patch_per_img = patch_per_img
        self.dataset_name = self.get_dataset_name()
        self.dataset = None

        self.load_dataset()

    def get_dataset_name(self):
        os.makedirs(name='datasets', exist_ok=True)
        dataset_name = self.subset + '_data_' + \
            str(self.patch_width) + 'x' + \
            str(self.patch_height) + 'ps_' + \
            str(len(self.scene_list)) + 'scenes_' + \
            str(self.patch_per_img) + 'ppi.pkl'
        return os.path.join('datasets', dataset_name)

    def load_dataset(self):
        if os.path.exists(self.dataset_name):
            print(self.dataset_name, ' exists')
            with open(self.dataset_name, 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            self.encode_to_pkl()

    def encode_to_pkl(self):
        print(self.subset, ':', self.dataset_name)

        if self.subset == 'train' or self.subset == 'valid':
            inputs = []
            targets = []
            for scene_name in self.scene_list:
                print('Processing scene ', scene_name)
                data_subdir = os.path.join(self.data_dir, scene_name)
                print('Visit data subdir ', data_subdir)
                for idx in range(60):
                    print('Encoding ', str(idx))
                    exr_name = str(idx) + '.exr'
                    albedo_path = os.path.join(data_subdir, 'albedo' + exr_name)
                    normal_path = os.path.join(data_subdir, 'shading_normal' + exr_name)
                    depth_path = os.path.join(data_subdir, 'depth' + exr_name)

                    noisy_path = os.path.join(data_subdir, 'color_accum' + exr_name)
                    real_path = os.path.join(data_subdir, 'reference' + exr_name)

                    # original albedo ranges between [0,1] ==> [0,1]
                    albedo = load_exr(albedo_path, datatype=np.float32, axis=0)
                    # original normal ranges between [-1,1] ==> [0,1]
                    normal = (load_exr(normal_path, datatype=np.float32, axis=0) + 1.0) * 0.5
                    # original depth ranges between [0,1] ==> [0,1]
                    depth = np.expand_dims((load_exr_one_channel(depth_path, datatype=np.float16)), axis=0)

                    # ? original noisy ranges between [0, infinity] ==> [0,1]
                    noisy = load_exr(noisy_path, datatype=np.float16, axis=0)
                    # ? original GT ranges between [0, infinity] ==> [0,1]
                    real = load_exr(real_path, datatype=np.float32, axis=0)

                    noisy_full_img = np.concatenate((noisy, albedo, normal, depth), axis=0)
                    noisy_full_img = noisy_full_img.astype(np.float32)
                    real_full_img = real.astype(np.float32)

                    for _ in range(self.patch_per_img):
                        noisy_one, target_one = self.random_crop(noisy_full_img, real_full_img)
                        aug_idx = np.random.randint(0, 8)
                        noisy_one = self.aug_input(noisy_one, aug_idx)
                        target_one = self.aug_input(target_one, aug_idx)
                        inputs.append(noisy_one)
                        targets.append(target_one)

            inputs = torch.from_numpy(np.stack(tuple(inputs), axis=0))
            targets = torch.from_numpy(np.stack(tuple(targets), axis=0))
            print(torch.mean(inputs))
            self.dataset = Data.TensorDataset(inputs, targets)

            print('save', self.subset, '...')
            with open(self.dataset_name, 'wb') as f:
                pickle.dump(self.dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            inputs = []
            targets = []
            for scene_name in self.scene_list:
                print('Processing scene ', scene_name)
                data_subdir = os.path.join(self.data_dir, scene_name)
                print('Visit data subdir ', data_subdir)
                for idx in range(60):
                    print('Encoding ', str(idx))
                    exr_name = str(idx) + '.exr'
                    albedo_path = os.path.join(data_subdir, 'albedo' + exr_name)
                    normal_path = os.path.join(data_subdir, 'shading_normal' + exr_name)
                    depth_path = os.path.join(data_subdir, 'depth' + exr_name)

                    noisy_path = os.path.join(data_subdir, 'color_accum' + exr_name)
                    real_path = os.path.join(data_subdir, 'reference' + exr_name)

                    # original albedo ranges between [0,1] ==> [0,1]
                    albedo = load_exr(albedo_path, datatype=np.float32, axis=0)
                    # original normal ranges between [-1,1] ==> [0,1]
                    normal = (load_exr(normal_path, datatype=np.float32, axis=0) + 1.0) * 0.5
                    # original depth ranges between [0,1] ==> [0,1]
                    depth = np.expand_dims((load_exr_one_channel(depth_path, datatype=np.float16)), axis=0)

                    # ? original noisy ranges between [0, infinity] ==> [0,1]
                    noisy = load_exr(noisy_path, datatype=np.float16, axis=0)
                    # ? original GT ranges between [0, infinity] ==> [0,1]
                    real = load_exr(real_path, datatype=np.float32, axis=0)

                    noisy_full_img = np.concatenate((noisy, albedo, normal, depth), axis=0)
                    noisy_full_img = noisy_full_img.astype(np.float16)
                    real_full_img = real.astype(np.float16)
                    inputs.append(noisy_full_img)
                    targets.append(real_full_img)

            inputs = torch.from_numpy(np.stack(tuple(inputs), axis=0))
            targets = torch.from_numpy(np.stack(tuple(targets), axis=0))
            self.dataset = Data.TensorDataset(inputs, targets)

            print('save', self.subset, '...')
            with open(self.dataset_name, 'wb') as f:
                pickle.dump(self.dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    def random_crop(self, x, y):
        offset_h, offset_w = self.generate_offset(x.shape)
        cropped_x = x[:, offset_h:offset_h + self.patch_height, offset_w:offset_w + self.patch_width]
        cropped_y = y[:, offset_h:offset_h + self.patch_height, offset_w:offset_w + self.patch_width]
        return cropped_x, cropped_y

    def generate_offset(self, shape):
        assert shape[1] >= self.patch_height
        assert shape[2] >= self.patch_width
        ch, h, w = shape
        range_h = h - self.patch_height
        range_w = w - self.patch_width
        offset_h = 0 if range_h == 0 else np.random.randint(range_h)
        if range_w == 0:
            offset_w = 0
        else:
            my_rand = np.random.randint(range_w)
            offset_w = 1 if my_rand == 0 else int(np.log2(my_rand) / np.log2(
                range_w) * range_w)
        return offset_h, offset_w

    @staticmethod
    def aug_input(img, idx=0):
        if idx == 0:
            return img
        elif idx == 1:
            return np.rot90(img, axes=(1, 2))
        elif idx == 2:
            return np.rot90(img, k=2, axes=(1, 2))  # 180
        elif idx == 3:
            return np.rot90(img, k=3, axes=(1, 2))  # 270
        elif idx == 4:
            return np.flipud(img)
        elif idx == 5:
            return np.flipud(np.rot90(img, axes=(1, 2)))
        elif idx == 6:
            return np.flipud(np.rot90(img, k=2, axes=(1, 2)))
        elif idx == 7:
            return np.flipud(np.rot90(img, k=3, axes=(1, 2)))


def assum_noisy(scene_name):
    print('process scene', scene_name)

    if scene_name == 'classroom':
        import data.classroom.camera_matrices as camera
    elif scene_name == 'livingroom':
        import data.livingroom.camera_matrices as camera
    elif scene_name == 'sanmiguel':
        import data.sanmiguel.camera_matrices as camera
    elif scene_name == 'sponza':
        import data.sponza.camera_matrices as camera
    elif scene_name == 'sponzaglossy':
        import data.sponzaglossy.camera_matrices as camera
    else:
        import data.sponzamovinglight.camera_matrices as camera

    camera_matrices = np.array(camera.camera_matrices)
    pixel_offsets = np.array(camera.pixel_offsets)
    position_limit_squared = camera.position_limit_squared
    normal_limit_squared = camera.normal_limit_squared
    second_blend_alpha = 0.1

    exr_path = '.exr'
    color_path = './data/' + scene_name + '/color'
    normal_path = './data/' + scene_name + '/shading_normal'
    position_path = './data/' + scene_name + '/world_position'
    save_path = './data/' + scene_name + '/color_accum'

    offsets = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        dtype=int
    )

    color0 = color_path + str(0) + exr_path
    normal0 = normal_path + str(0) + exr_path
    position0 = position_path + str(0) + exr_path

    color0 = load_exr(color0, datatype=np.float32)
    normal0 = load_exr(normal0, datatype=np.float32)
    position0 = load_exr(position0, datatype=np.float32)

    print('save 0 exr')
    save_exr(color0, save_path + str(0) + exr_path)

    h, w, _ = position0.shape
    ones = np.ones((h, w, 1))
    prev_spp = np.ones((h, w))

    for f in range(1, 60):
        print('process exr', f)

        camera_matrix = camera_matrices[f - 1]
        pixel_offset = pixel_offsets[f - 1]

        color1 = color_path + str(f) + exr_path
        normal1 = normal_path + str(f) + exr_path
        position1 = position_path + str(f) + exr_path

        color1 = load_exr(color1, datatype=np.float32)
        normal1 = load_exr(normal1, datatype=np.float32)
        position1 = load_exr(position1, datatype=np.float32)

        prev_frame = np.dot(np.append(position1, ones, axis=2), camera_matrix)
        prev_frame_uv = prev_frame[:, :, 0:2] / prev_frame[:, :, 3:4]
        prev_frame_uv = (prev_frame_uv + 1) / 2
        prev_frame_pixel_f = prev_frame_uv * (w, h)
        prev_frame_pixel_f = prev_frame_pixel_f - (pixel_offset[0], 1 - pixel_offset[1])
        prev_frame_pixel = np.floor(prev_frame_pixel_f).astype(int)

        prev_pixel_fract = prev_frame_pixel_f - prev_frame_pixel
        one_minus_prev_pixel_fract = 1 - prev_pixel_fract

        weight0 = one_minus_prev_pixel_fract[:, :, 0:1] * one_minus_prev_pixel_fract[:, :, 1:2]
        weight1 = prev_pixel_fract[:, :, 0:1] * one_minus_prev_pixel_fract[:, :, 1:2]
        weight2 = one_minus_prev_pixel_fract[:, :, 0:1] * prev_pixel_fract[:, :, 1:2]
        weight3 = prev_pixel_fract[:, :, 0:1] * prev_pixel_fract[:, :, 1:2]
        weight = np.concatenate((weight0, weight1, weight2, weight3), axis=2)

        current_spp = np.ones((h, w))
        new_color = np.zeros((h, w, 3))

        for i in range(h):
            for j in range(w):
                total_weight = 0
                sample_spp = 0
                prev_color = np.zeros(3)
                blend_alpha = 1
                pixel_position = prev_frame_pixel[i, j]

                for k in range(4):
                    x, y = pixel_position + offsets[k]

                    if 0 <= x < w and 0 <= y < h:
                        position_difference = position0[y, x] - position1[i, j]
                        position_difference_squared = np.inner(position_difference, position_difference)

                        if position_difference_squared < position_limit_squared:
                            normal_difference = normal0[y, x] - normal1[i, j]
                            normal_difference_squared = np.inner(normal_difference, normal_difference)

                            if normal_difference_squared < normal_limit_squared:
                                sample_spp += weight[i, j, k] * prev_spp[y, x]
                                prev_color += weight[i, j, k] * color0[y, x]
                                total_weight += weight[i, j, k]

                if total_weight > 0:
                    prev_color /= total_weight
                    sample_spp /= total_weight
                    blend_alpha = 1 / (1 + sample_spp)
                    blend_alpha = max(blend_alpha, second_blend_alpha)

                new_spp = 1
                if blend_alpha < 1:
                    if sample_spp >= 255:
                        new_spp = 255
                    else:
                        new_spp = round(sample_spp)
                current_spp[i, j] = new_spp
                new_color[i, j] += blend_alpha * color1[i, j] + (1 - blend_alpha) * prev_color

        print('save exr', f)
        save_exr(new_color, save_path + str(f) + exr_path)

        color0 = color1
        normal0 = normal1
        position0 = position1
        prev_spp = current_spp


def get_depth(scene_name):
    print('process scene', scene_name)

    if scene_name == 'classroom':
        import data.classroom.camera_matrices as camera
    elif scene_name == 'livingroom':
        import data.livingroom.camera_matrices as camera
    elif scene_name == 'sanmiguel':
        import data.sanmiguel.camera_matrices as camera
    elif scene_name == 'sponza':
        import data.sponza.camera_matrices as camera
    elif scene_name == 'sponzaglossy':
        import data.sponzaglossy.camera_matrices as camera
    else:
        import data.sponzamovinglight.camera_matrices as camera

    exr_path = '.exr'
    position_path = './data/' + scene_name + '/world_position'
    save_path = './data/' + scene_name + '/depth'

    camera_matrices = np.asarray(camera.camera_matrices)
    for i in range(60):
        print('process exr', i)
        camera_matrix = np.asarray(camera_matrices[i])
        position = load_exr(position_path + str(i) + exr_path, datatype=np.float32)
        h, w, _ = position.shape
        ones = np.ones((h, w, 1))
        prev_frame = np.dot(np.append(position, ones, axis=2), camera_matrix)
        prev_frame = prev_frame[:, :, 2] / prev_frame[:, :, 3] - 1

        print('save exr', i)
        save_exr_one_channel(prev_frame, save_path + str(i) + exr_path)


if __name__ == '__main__':
    scenes = ['classroom', 'livingroom', 'sanmiguel', 'sponza', 'sponzaglossy', 'sponzamovinglight']
    for scene in scenes:
        get_depth(scene)

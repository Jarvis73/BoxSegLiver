import numpy as np
import random
from scipy.ndimage import label, find_objects
from batchgenerators.transforms import AbstractTransform
from batchgenerators.dataloading import DataLoader
from batchgenerators.dataloading.dataset import Dataset
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk
from collections import OrderedDict


def compute_robust_moments(binary_image, isotropic=False, indexing="ij", min_std=1.0, spacing_after_resampling=None):
    """
    Compute robust center and standard deviation of a binary image(0: background, 1: foreground).

    Support n-dimension array.

    Parameters
    ----------
    binary_image: ndarray
        Input a image
    isotropic: boolean
        Compute isotropic standard deviation or not.
    indexing: {'xy', 'ij'}, optional
        Cartesian ('xy') or matrix ('ij', default) indexing of output.
        See Notes for more details.
    min_std: float
        Set stddev lower bound

    Returns
    -------
    center: ndarray
        A vector with dimension = `binary_image.ndim`. Median of all the points assigned 1.
    std_dev: ndarray
        A vector with dimension = `binary_image.ndim`. Standard deviation of all the points assigned 1.

    Notes
    -----
    All the points assigned 1 are considered as a single object.

    """
    #
    ndim = binary_image.ndim
    coords = np.nonzero(binary_image)
    points = np.asarray(coords).astype(np.float32)
    #无用户标记，返回（-1，-1，-1）
    if points.shape[1] == 0:
        return np.array([-1.0] * ndim, dtype=np.float32), np.array([-1.0] * ndim, dtype=np.float32)
    #变成坐标，计算中心
    points = np.transpose(points)       # [pts, 2], 2: (i, j)
    center = np.median(points, axis=0)  # [2]

    # Compute median absolute deviation(short for mad) to estimate standard deviation
    #是否分长短轴,不分长短轴需要
    if isotropic:
        diff = np.linalg.norm(points - center, axis=1)
        mad = np.median(diff)
        if spacing_after_resampling is None:
            mad = np.array([mad] * ndim)
        else:
            mad = spacing_after_resampling * mad / (np.max(spacing_after_resampling) + 0.05)
    else:
        diff = np.absolute(points - center)
        mad = np.median(diff, axis=0)
    std_dev = 1.4826 * mad
    std_dev = np.maximum(std_dev, [min_std] * ndim)
    if not indexing or indexing == "xy":
        return center[::-1], std_dev[::-1]
    elif indexing == "ij":
        return center, std_dev
    else:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")


def get_center(binary_image, indexing="ij"):
    #
    ndim = binary_image.ndim
    coords = np.nonzero(binary_image)
    points = np.asarray(coords).astype(np.float32)
    #无用户标记，返回（-1，-1，-1）
    if points.shape[1] == 0:
        return np.array([-1.0] * ndim, dtype=np.float32), np.array([-1.0] * ndim, dtype=np.float32)
    #变成坐标，计算中心
    points = np.transpose(points)       # [pts, 2], 2: (i, j)
    center = np.median(points, axis=0)  # [2]
    if not indexing or indexing == 'xy':
        return center[::-1]
    elif indexing == "ij":
        return center
    else:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")


def get_centers_from_label(labels, which_class=2, indexing="ij"):
    if not np.any(labels == which_class):
        return []

    # 找一下连通类
    obj_lab, num_objects = label((labels == which_class).astype(int))

    center_p_list = []
    for o in range(1, num_objects + 1):
        center = get_center(obj_lab == o, indexing=indexing)
        center_p_list.append(center)

    return center_p_list


def create_gaussian_distribution(shape, centers, stddevs, indexing="ij", keepdims=False):
    """
    Parameters
    ----------
    #shape: list
        two values
    centers: ndarray
        Float ndarray with shape [n, d], d means (x, y, ...)
    stddevs: ndarray
        Float ndarray with shape [n, d], d means (x, y, ...)
    indexing: {'xy', 'ij'}, optional
        Cartesian ('xy') or matrix ('ij', default) indexing of output.
        See Notes for more details.
    keepdims: bool
        Keep final dimension

    TODO(zjw) Warning: indexing='xy' need test

    Returns
    -------
    A batch of spatial guide image with shape [h, w, 1] for 2D and [d, h, w, 1] for 3D

    Notes
    -----
    -1s in center and stddev are padding value and almost don't affect spatial guide
    """
    #n个中心, 最后取各点处最大的值，最为最终guide
    centers = np.asarray(centers, np.float32)                                   # [n, 3]    if dimension=3
    stddevs = np.asarray(stddevs, np.float32)                                   # [n, 3]
    assert centers.ndim == 2, centers.shape
    assert centers.shape == stddevs.shape, (centers.shape, stddevs.shape)
    coords = [np.arange(0, s) for s in shape]
    coords = np.tile(
        np.stack(np.meshgrid(*coords, indexing=indexing), axis=-1)[None],
        reps=[centers.shape[0]] + [1] * (centers.shape[1] + 1))                 # [n, h, w, d, 3]
    coords = coords.astype(np.float32)
    centers = centers[..., None, None, None, :]                                       # [n, 1, 1, 1, 3]
    stddevs = stddevs[..., None, None, None, :]                                       # [n, 1, 1, 1, 3]
    normalizer = 2 * stddevs * stddevs                                          # [n, 1, 1, 1, 3]
    d = np.exp(-np.sum((coords - centers) ** 2 / normalizer, axis=-1, keepdims=keepdims))   # [n, h, w, d, 1] / [n, h, w, d]
    return np.max(d, axis=0)                                                    # [h, w, d, 1] / [h, w, d]


def create_binary_ball_init(shape, centers, stddevs, indexing='ij', keepdims=False):
    centers = np.asarray(centers, np.float32)  # [n, 3]    if dimension=3
    stddevs = np.asarray(stddevs, np.float32)  # [n, 3]
    assert centers.ndim == 2, centers.shape
    assert centers.shape == stddevs.shape, (centers.shape, stddevs.shape)
    coords = [np.arange(0, s) for s in shape]
    coords = np.tile(
        np.stack(np.meshgrid(*coords, indexing=indexing), axis=-1)[None],
        reps=[centers.shape[0]] + [1] * (centers.shape[1] + 1))  # [n, h, w, d, 3]
    coords = coords.astype(np.float32)
    centers = centers[..., None, None, None, :]  # [n, 1, 1, 1, 3]
    stddevs = stddevs[..., None, None, None, :]  # [n, 1, 1, 1, 3]
    normalizer = stddevs * stddevs / 1.48 / 1.48  # [n, 1, 1, 1, 3]
    d = (np.sum((coords - centers) ** 2 / normalizer, axis=-1, keepdims=keepdims) < 1).astype(int) # [n, h, w, d, 1] / [n, h, w, d]
    return np.max(d, axis=0)  # [h, w, d, 1] / [h, w, d]


def get_gd_image_single_object(labels, center_perturb=0.2, stddev_perturb=0.4, blank_prob=0.5, which_class=2,
                               isotropic=True, partial=False, partial_slice="first", min_std=1.0, indexing="ij",
                               keepdims=False, random_drop_componets_prob=0.0, spacing_after_resampling=None):
    """
        Get gaussian distribution image with some perturbation. All points assigned 1 are considered
        to be the same object.

        Support n-dimension array.

        Parameters
        ----------
        labels: ndarray
            A binary image
        center_perturb: float
            Position perturbation of central point
        stddev_perturb: float
            Scale perturbation of standard deviation
        blank_prob: float
            Probability of returning a blank image(which means no gaussian distribution guide)
        partial: bool
            For test mode. If true only first slice has spatial guide for each tumor
        partial_slice: str
            Which slice to annotate spatial guide when partial is True. ["first", "middle"] are supported.
        only_moments: bool
            Only return moments
        min_std: float
            Set stddev lower bound
        indexing: {'xy', 'ij'}, optional
            Cartesian ('xy') or matrix ('ij', default) indexing of output.
            See Notes for more details.
        keepdims: bool
            Keep final dimension

        Returns
        -------
        gd: ndarray
            Gaussian distribution image

        """
    # if partial_slice not in ["first", "middle"]:
    #     raise ValueError("Only support `first` and `middle`, got {}".format(partial_slice))

    ndim = labels.ndim

    if not np.any(labels == which_class) or random.random() < blank_prob or which_class == -1:
        # return a blank gd image
        #print('blank_prob')
        #print('blank_shape: ', labels.shape)
        return np.zeros(labels.shape)# + 0.5

    #找一下连通类
    obj_lab, num_objects = label((labels == which_class).astype(int))
    obj_ndim = ndim

    center_p_list = []
    std_p_list = []
    for o in range(1, num_objects + 1):
        if random.random() < random_drop_componets_prob:
            continue
        center, std = compute_robust_moments(obj_lab == o, isotropic=isotropic, indexing=indexing, min_std=min_std,
                                             spacing_after_resampling=spacing_after_resampling)
        center_p_ratio = np.random.uniform(-center_perturb, center_perturb, obj_ndim)
        center_p_list.append(center_p_ratio * std + center)
        std_p_ratio = np.random.uniform(1.0 / (1 + stddev_perturb), 1.0 + stddev_perturb, obj_ndim)
        std_p_list.append(std_p_ratio * std)
    #print("do_guass...")
    #print("connected components", num_objects)
    #print("center:", center_p_list)
    #print("std:", std_p_list)
    if len(center_p_list) == 0:
        return np.zeros(labels.shape)

    cur_gd = create_gaussian_distribution(obj_lab.shape, center_p_list, std_p_list, indexing=indexing, keepdims=keepdims)
    return cur_gd# * 0.5 +0.5


def get_gd_image_single_object_click_ball(labels, center_perturb=0.2, stddev_perturb=0.4, blank_prob=0.5, which_class=2,
                                          min_std=3.0, indexing="ij", keepdims=False, random_drop_componets_prob=0.0):

    ndim = labels.ndim

    if not np.any(labels == which_class) or random.random() < blank_prob or which_class == -1:
        return np.zeros(labels.shape)

    # 找一下连通类
    obj_lab, num_objects = label((labels == which_class).astype(int))
    obj_ndim = ndim

    center_p_list = []
    std_p_list = []
    for o in range(1, num_objects + 1):
        if random.random() < random_drop_componets_prob:
            continue
        center = get_center(obj_lab == o, indexing=indexing)
        std = np.array([min_std] * obj_ndim)
        center_p_ratio = np.random.uniform(-center_perturb, center_perturb, obj_ndim)
        center_p_list.append(center_p_ratio * std + center)
        std_p_ratio = np.random.uniform(1.0 / (1 + stddev_perturb), 1.0 + stddev_perturb, obj_ndim)
        std_p_list.append(std_p_ratio * std)
    # print("do_guass...")
    # print("connected components", num_objects)
    # print("center:", center_p_list)
    # print("std:", std_p_list)
    if len(center_p_list) == 0:
        return np.zeros(labels.shape)

    cur_gd = create_gaussian_distribution(obj_lab.shape, center_p_list, std_p_list, indexing=indexing,
                                          keepdims=keepdims)
    return cur_gd  # * 0.5 +0.5


def get_gd_image_single_object_binary(labels, center_perturb=0.2, stddev_perturb=0.4, blank_prob=0.5, which_class=2,
                                      isotropic=True, partial=False, partial_slice="first", min_std=1.0, indexing="ij",
                                      keepdims=False, random_drop_componets_prob=0.0, spacing_after_resampling=None):

    ndim = labels.ndim

    if not np.any(labels == which_class) or random.random() < blank_prob or which_class == -1:
        return np.zeros(labels.shape)

    # 找一下连通类
    obj_lab, num_objects = label((labels == which_class).astype(int))
    obj_ndim = ndim

    center_p_list = []
    std_p_list = []
    for o in range(1, num_objects + 1):
        if random.random() < random_drop_componets_prob:
            continue
        center, std = compute_robust_moments(obj_lab == o, isotropic=isotropic, indexing=indexing, min_std=min_std,
                                             spacing_after_resampling=spacing_after_resampling)
        center_p_ratio = np.random.uniform(-center_perturb, center_perturb, obj_ndim)
        center_p_list.append(center_p_ratio * std + center)
        std_p_ratio = np.random.uniform(1.0 / (1 + stddev_perturb), 1.0 + stddev_perturb, obj_ndim)
        std_p_list.append(std_p_ratio * std)

    if len(center_p_list) == 0:
        return np.zeros(labels.shape)

    cur_init = create_binary_ball_init(obj_lab.shape, center_p_list, std_p_list, indexing=indexing, keepdims=keepdims)

    return cur_init


class Spatial_Guide_Augment(AbstractTransform):
    def __init__(self, which_class=-1, blank_prob=0.5, center_perturb=0.2, stddev_perturb=0.4, isotropic=False,
                 data_key='data', seg_key='target', random_drop_components_prob=0.0):
        self.center_perturb = center_perturb
        self.stddev_perturb = stddev_perturb
        self.data_key = data_key
        self.seg_key = seg_key
        self.which_class = which_class
        self.blank_prob = blank_prob
        self.isotropic = isotropic
        self.random_drop_components_prob = random_drop_components_prob

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.seg_key)
        #print('original_data_shape: ', data.shape)
        #print('original_seg_shape: ', seg.shape)
        spatial_guide = np.zeros(seg.shape, dtype=seg.dtype)
        for b in range(spatial_guide.shape[0]):
            # spatial_guide[b, 0] = get_gd_image_single_object(seg[b, 0], which_class=self.which_class,
            #                                                  blank_prob=self.blank_prob,
            #                                                  center_perturb=self.center_perturb,
            #                                                  stddev_perturb=self.stddev_perturb,
            #                                                  isotropic=self.isotropic,
            #                                                  random_drop_componets_prob=self.random_drop_components_prob)
            spatial_guide[b, 0] = get_gd_image_single_object_click_ball(seg[b, 0], which_class=self.which_class,
                                                                        blank_prob=self.blank_prob,
                                                                        center_perturb=self.center_perturb,
                                                                        stddev_perturb=self.stddev_perturb,
                                                                        min_std=3.0,
                                                                        random_drop_componets_prob=self.random_drop_components_prob)
        data = np.concatenate((data, spatial_guide), axis=1)
        data_dict[self.data_key] = data
        #print('now_data_shape: ', data.shape)
        #print('now_seg_shape: ', seg.shape)
        return data_dict


def tumor_fn_spatial_guide(seg, labels, center_perturb=0.0, stddev_perturb=0.0, blank_prob=0.0, which_class=2,
                           isotropic=False, min_std=3.0, spacing_after_resampling=None, just_return_label=False):
    assert seg.shape == labels.shape, "seg shape and labels shape must be the same!"
    #制作spatial_guide_label
    spatial_guide_label = np.zeros(seg.shape)
    obj_lab, num_objects = label((labels == which_class).astype(int))
    components = find_objects(obj_lab)
    count = 0
    for i, slicer in enumerate(components):
        obj_mask = (obj_lab[slicer] == (i + 1))
        voxel_intersect = (seg[slicer][obj_mask] == which_class).sum()
        voxel_labels = obj_mask.sum()
        if voxel_intersect / voxel_labels < 0.25:
            spatial_guide_label[slicer][obj_mask] = which_class
            count += 1

    if just_return_label:
        return spatial_guide_label, count

    spatial_guide = get_gd_image_single_object(spatial_guide_label, center_perturb=center_perturb,
                                               stddev_perturb=stddev_perturb, blank_prob=blank_prob,
                                               which_class=which_class, isotrpoic=isotropic, min_std=min_std,
                                               spacing_after_resampling=spacing_after_resampling)

    return spatial_guide


def tumor_fp_spatial_guide(seg, labels, center_perturb=0.0, stddev_perturb=0.0, blank_prob=0.0, which_class=2,
                           isotropic=False, min_std=3.0, spacing_after_resampling=None, just_return_label=False):
    assert seg.shape == labels.shape, "seg shape and labels shape must be the same!"
    #制作spatial_guide_label
    spatial_guide_label = np.zeros(seg.shape)
    obj_lab, num_objects = label((labels == which_class).astype(int))
    components = find_objects(obj_lab)
    count = 0
    for i, slicer in enumerate(components):
        obj_mask = (obj_lab[slicer] == (i + 1))
        voxel_intersect = (seg[slicer][obj_mask] == which_class).sum()
        if voxel_intersect == 0:
            spatial_guide_label[slicer][obj_mask] = which_class
            count += 1

    if just_return_label:
        return spatial_guide_label, count

    spatial_guide = get_gd_image_single_object(spatial_guide_label, center_perturb=center_perturb,
                                               stddev_perturb=stddev_perturb, blank_prob=blank_prob,
                                               which_class=which_class, isotrpoic=isotropic, min_std=min_std,
                                               spacing_after_resampling=spacing_after_resampling)

    return spatial_guide


def only_keep_the_largest_connected_components(predicted_segmentation, **_):
    print('only keep the largest connected component...')
    obj_pred, obj_num = label((predicted_segmentation > 0).astype(int))
    sizes = []
    for o in range(1, obj_num + 1):
        sizes.append((obj_pred == o).sum())
    mx = np.argmax(sizes) + 1
    print(sizes)
    predicted_segmentation[obj_pred != mx] = 0
    return predicted_segmentation


def default_collate(batch):
    '''
    heavily inspired by the default_collate function of pytorch
    :param batch:
    :return:
    '''
    if isinstance(batch[0], np.ndarray):
        return np.vstack(batch)
    elif isinstance(batch[0], (int, np.int64)):
        return np.array(batch).astype(np.int32)
    elif isinstance(batch[0], (float, np.float32)):
        return np.array(batch).astype(np.float32)
    elif isinstance(batch[0], (np.float64,)):
        return np.array(batch).astype(np.float64)
    elif isinstance(batch[0], (dict, OrderedDict)):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    elif isinstance(batch[0], str):
        return batch
    else:
        raise TypeError('unknown type for batch:', type(batch))


class DataLoader_3d_inference(DataLoader):
    def __init__(self, data, batch_size, num_threads_in_multithreaded, seed_for_shuffle=1, collate_fn=default_collate,
                 return_incomplete=False, shuffle=True, infinite=False):
        '''
        A simple dataloader that can take a Dataset as data.
        It is not super efficient because I cannot make too many hard assumptions about what data_dict will contain.
        If you know what you need, implement your own!
        :param data:
        :param batch_size:
        :param num_threads_in_multithreaded:
        :param seed_for_shuffle:
        '''
        super(DataLoader_3d_inference, self).__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle,
                                                    return_incomplete=return_incomplete, shuffle=shuffle,
                                                    infinite=infinite)
        self.collate_fn = collate_fn
        assert isinstance(self._data, Dataset)
        self.indices = np.arange(len(data))

    def generate_train_batch(self):
        indices = self.get_indices()

        batch = [self._data[i] for i in indices]

        return self.collate_fn(batch)


def get_tmuor_fn_label_from_dir(input_dir_seg, input_dir_label, output_dir):
    maybe_mkdir_p(output_dir)
    nifti_files_seg = subfiles(input_dir_seg, suffix='nii', join=False)
    nifti_files_label = subfiles(input_dir_label, suffix='nii.gz', join=False)
    nifti_files_seg.remove('test-segmentation-59.nii')
    print(nifti_files_seg, '\n', len(nifti_files_seg))
    print(nifti_files_label, '\n', len(nifti_files_label))

    info = []
    for seg_fname, label_fname in zip(nifti_files_seg, nifti_files_label):
        print('\n', seg_fname)
        identifier = str(seg_fname.split("-")[-1][:-4])
        output_fname = join(output_dir, "test_fn_%s.nii.gz" % identifier)
        seg_itk = sitk.ReadImage(join(input_dir_seg, seg_fname))
        seg = sitk.GetArrayFromImage(seg_itk)
        label_itk = sitk.ReadImage(join(input_dir_label, label_fname))
        label = sitk.GetArrayFromImage(label_itk)
        label[label == 1] = 2
        assert seg.shape == label.shape, (seg.shape, label.shape)
        spatial_guide_label, count = tumor_fn_spatial_guide(seg, label, just_return_label=True)
        print("# of FN: ", count)
        info.append(OrderedDict())
        info[-1]["filename"] = "test_fn_%s.nii.gz" % identifier
        info[-1]["fn_num"] = count
        info[-1]["centers"] = get_centers_from_label(spatial_guide_label)
        print("centers of FN:\n", info[-1]["centers"], '\n')
        fn = sitk.GetImageFromArray(spatial_guide_label)
        fn.CopyInformation(seg_itk)
        sitk.WriteImage(fn, output_fname)

    f = open('./fn_info.pkl', 'wb')
    pickle.dump(info, f)


def get_tmuor_fp_label_from_dir(input_dir_seg, input_dir_label, output_dir):
    maybe_mkdir_p(output_dir)
    nifti_files_seg = subfiles(input_dir_seg, suffix='nii', join=False)
    nifti_files_label = subfiles(input_dir_label, suffix='nii.gz', join=False)
    #nifti_files_seg.remove('test-segmentation-59.nii')
    print(nifti_files_seg, '\n', len(nifti_files_seg))
    print(nifti_files_label, '\n', len(nifti_files_label))

    info = []
    for seg_fname, label_fname in zip(nifti_files_seg, nifti_files_label):
        print('\n', seg_fname)
        identifier = str(seg_fname.split("-")[-1][:-4])
        output_fname = join(output_dir, "test_fp_%s.nii.gz" % identifier)
        seg_itk = sitk.ReadImage(join(input_dir_seg, seg_fname))
        seg = sitk.GetArrayFromImage(seg_itk)
        label_itk = sitk.ReadImage(join(input_dir_label, label_fname))
        label = sitk.GetArrayFromImage(label_itk)
        label[label == 1] = 2
        seg[seg < 2] = 0
        assert seg.shape == label.shape, (seg.shape, label.shape)
        spatial_guide_label, count = tumor_fp_spatial_guide(label, seg, just_return_label=True)
        print("# of FP: ", count)
        info.append(OrderedDict())
        info[-1]["filename"] = "test_fp_%s.nii.gz" % identifier
        info[-1]["fn_num"] = count
        info[-1]["centers"] = get_centers_from_label(spatial_guide_label)
        print("centers of FP:\n", info[-1]["centers"], '\n')
        fn = sitk.GetImageFromArray(spatial_guide_label)
        fn.CopyInformation(seg_itk)
        sitk.WriteImage(fn, output_fname)

    f = open('./fp_info.pkl', 'wb')
    pickle.dump(info, f)


if __name__ == '__main__':
    print("running ...")
    input_dir_seg = 'E:/Dataset/LiTS/merge_013_gnet_sp_rand'
    input_dir_label = r'D:\0WorkSpace\MedicalImageSegmentation\data\LiTS\Test_Label'
    output_dir = r'D:\0WorkSpace\MedicalImageSegmentation\data\LiTS\aaa'
    # get_tmuor_fp_label_from_dir(input_dir_seg, input_dir_label, output_dir)
    get_tmuor_fn_label_from_dir(input_dir_seg, input_dir_label, output_dir)

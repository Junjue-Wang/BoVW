# Date: 2019.08.23
# Author: Kingdrone
import numpy as np
from skimage.io import imread
import os
import glob
from skimage.feature import greycomatrix, greycoprops
from sklearn.cluster import KMeans
import logging
from sklearn.externals import joblib
from tqdm import tqdm
import json


SEED = 2333
class SampleGenerator(object):
    def __init__(self, root_dir=r'../data/12class_tif', kernel_size=32, stride=16, extractors=None):
        super(SampleGenerator, self).__init__()
        self.cls_list = []
        self.imgp_list = []
        for idx, (root, dirs, files) in enumerate(os.walk(root_dir, topdown=False)):
            for name in files:
                img_path = os.path.join(root, name)
                self.imgp_list.append(img_path)
                self.cls_list.append(idx)
        self._slice_list = list(self._generate_slices(kernel_size, stride))
        self.extractors = extractors
        self.k_means_dict = dict()
        self.features_list = []

    def k_fold(self, k_fold=5, k_for_test=1):
        n = len(self.cls_list)
        fold_size = n // k_fold
        last_fold_size = n - fold_size * k_fold + fold_size

        indices = np.array(list(range(n)))
        rs = np.random.RandomState(SEED)
        rs.shuffle(indices)
        indices = list(indices)
        k_fold_indices = []
        for i in range(k_fold - 1):
            sub_inds = indices[i * fold_size:i * fold_size + fold_size]
            k_fold_indices.append(sub_inds)
        k_fold_indices.append(indices[(k_fold - 1) * fold_size: (k_fold - 1) * fold_size + last_fold_size])

        test_inds = k_fold_indices.pop(k_for_test - 1)
        train_inds = []
        for indices in k_fold_indices:
            train_inds += indices
        assert len(train_inds) + len(test_inds) == n

        return train_inds, test_inds

    def _generate_slices(self, kernel_size, stride_size):
        x = np.arange(0, 200, kernel_size-stride_size)
        x1s, y1s = np.meshgrid(x, x)

        for x1, y1 in zip(x1s.ravel(), y1s.ravel()):
            if x1 + kernel_size < 200 and y1 + kernel_size < 200:
                yield x1, y1, x1 + kernel_size, y1 + kernel_size

    def get_image_patches(self, idx):
        image_patches = []
        image_np = imread(self.imgp_list[idx])
        for region in self._slice_list:
            x1, y1, x2, y2 = region
            patch_np = image_np[x1:x2, y1:y2, :]
            image_patches.append(patch_np)
        return image_patches

    @property
    def image_num(self):
        return len(self.cls_list)

    def generate_container(self):
        features_container = dict()
        for ext in self.extractors:
            features_container[ext.__class__.__name__] = []
        return features_container

    @staticmethod
    def save_np(log_dir, feature_dict):
        np.save(os.path.join(log_dir, feature_dict['filename'].replace('tif', 'npy')), feature_dict)

    def generate_cluster_n(self, n_list=[], log_dir=r'../log', restore=True):
        if restore:
            print('**Restore Features!**')
            features_plist = glob.glob(os.path.join(log_dir, '*.npy'))
            for feature_path in tqdm(features_plist):
                self.features_list.append(np.load(feature_path).item())
            print('**Restore K-means!**')
            # self.cluster_n(n_list=n_list)
            for ext in tqdm(self.extractors):
                self.k_means_dict[ext.__class__.__name__] = joblib.load(ext.__class__.__name__)

        else:
            assert len(n_list) == len(self.extractors)

            print('**Extract Features!**')
            for i in tqdm(range(self.image_num)):
                image_np = imread(self.imgp_list[i])
                feature_dict = dict(
                    filename=str(self.cls_list[i]) + '_' + os.path.basename(self.imgp_list[i]),
                    features=self.generate_container()
                )
                for patch_i, region in enumerate(self._slice_list):
                    x1, y1, x2, y2 = region
                    patch_np = image_np[x1:x2, y1:y2, :]
                    for ext in self.extractors:
                        # features[ext.__class__.__name__].append(ext(patch_np))
                        feature_dict['features'][ext.__class__.__name__].append(ext(patch_np))
                self.features_list.append(feature_dict)
                self.save_np(log_dir, feature_dict)
                # print('Clustering: %d/%d' % (i+1, self.image_num))
            print('**Cluster K-means!**')

            self.cluster_n(n_list=n_list)
            # for idx, (k, v) in enumerate(features.items()):
            #     num_patches = len(v)
            #     k_means = KMeans(init='k-means++', n_clusters=n_list[idx], n_init=10)
            #     k_means.fit(np.array(v).reshape(num_patches, -1))
            #     joblib.dump(k_means, k)
            #     self.k_means_dict[k] = k_means

    def cluster_n(self, n_list=[]):
        # k means
        features = self.generate_container()
        for feat in self.features_list:
            for ext in self.extractors:
                features[ext.__class__.__name__].append(feat['features'][ext.__class__.__name__])

        for idx, (k, v) in enumerate(features.items()):
            k_means = KMeans(init='k-means++', n_clusters=n_list[idx], n_init=10)
            k_means.fit(np.array(v).reshape(-1, self.extractors[idx].dim))
            joblib.dump(k_means, k)
            self.k_means_dict[k] = k_means

    @property
    def _n_clusters(self):
        n_list = []
        for v in self.k_means_dict.values():
            n_list.append(v.n_clusters)
        return n_list

    def generate_frequency(self, log_dir=r'../log'):
        assert len(self.features_list) != 0
        start_feat_idx = 0
        for idx, (k, v) in enumerate(self.k_means_dict.items()):
            print('**Generate Frequency % s**' % k)
            for feat in tqdm(self.features_list):
                if idx == 0:
                    feat['frequency'] = [0] * sum(self._n_clusters)
                res = v.predict(np.array(feat['features'][k]))
                for i in set(res):
                    feat['frequency'][start_feat_idx+i] = list(res).count(i)
                self.save_np(log_dir, feat)
            start_feat_idx += self._n_clusters[idx]
                    # res = v.predict(f_i)

    def dataset_generator(self, n_list=[], k_fold=5, k_for_test=1, restore=True, log_dir=r'../log'):
        self.generate_cluster_n(n_list=n_list, restore=restore, log_dir=log_dir)
        self.generate_frequency(log_dir=log_dir)
        train_inds, test_inds = self.k_fold(k_fold, k_for_test)
        train_samples = np.array([self.features_list[train_id]['frequency'] for train_id in train_inds])
        train_labels = np.array([self.cls_list[train_id] for train_id in train_inds])
        test_samples = np.array([self.features_list[test_id]['frequency'] for test_id in test_inds])
        test_labels = np.array([self.cls_list[test_id] for test_id in test_inds])
        return train_samples, train_labels, test_samples, test_labels

class MeanSTD(object):
    def __init__(self):
        self.dim = 6
    def __call__(self, img_patch):
        tpatch = img_patch.copy().reshape(-1, 3)
        mean = np.mean(tpatch, 0)
        std = np.std(tpatch, 0)
        res = np.hstack((mean, std))
        self.dim = res.shape[0]
        return res

class GLCM(object):
    def __init__(self, distance:list=[1], angles=[0, np.pi/4, np.pi/2, np.pi*3/4], levels=64):
        self.distance = distance
        self.angles = angles
        self.levels = levels
        self.dim=12
    def __call__(self, img_patch):
        channel = img_patch.shape[-1]
        correlation_list = []
        homogeneity_list = []
        ASM_list = []
        contrast_list = []
        for i in range(channel):
            compressed_path_i = img_patch[:, :, i] / 256 * self.levels
            glcm = greycomatrix(compressed_path_i.astype(np.uint8), self.distance, self.angles, self.levels)
            # dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
            correlation_list.append(greycoprops(glcm, 'correlation')[0, 0])
            homogeneity_list.append(greycoprops(glcm, 'homogeneity')[0, 0])
            ASM_list.append(greycoprops(glcm, 'ASM')[0, 0])
            contrast_list.append(greycoprops(glcm, 'contrast')[0, 0])

        res = np.hstack((correlation_list, homogeneity_list, ASM_list, contrast_list))
        self.dim = res.shape[0]
        return res


if __name__ == '__main__':
    pg = SampleGenerator(root_dir=r'C:\Users\mi\Desktop\作业\Google dataset of SIRI-WHU_earth_im_tiff\12class_tif', extractors=[MeanSTD(), GLCM()])
    train_samples, train_labels, test_samples, test_labels = pg.dataset_generator([30, 30], restore=True)
# if __name__ == '__main__':
#     ps = PatchGenerator()
#     f = MeanSTD()
#     t = ps.get_image_patches(5)[0]
#     a = f(t)
#     print(a.shape)
    # img = imread(r'D:\WorkSpace\BoVW\data\12class_tif\agriculture\0001.tif')[64:128, 64:128, 2] / 256 * 8
    # print(img.astype(np.uint8))
    # glcm = greycomatrix(img.astype(np.uint8), [3], [0, np.pi/4, np.pi/2, np.pi*3/4], 8, symmetric=True, normed=True)
    # correlation = greycoprops(glcm, 'correlation')[0, 0]
    # homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    # ASM = greycoprops(glcm, 'ASM')[0, 0]
    # contrast = greycoprops(glcm, 'contrast')[0, 0]
    # print(correlation)
    # print(homogeneity)
    # print(contrast)

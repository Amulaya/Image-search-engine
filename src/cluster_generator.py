import shutil
from pathlib import Path
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import glob

from tqdm import tqdm


class ClusterGenerator:
    def execute(self, dataset_dir_path: Path, results_dir_path: Path):
        image.LOAD_TRUNCATED_IMAGES = True
        model = VGG16(weights='imagenet', include_top=False)

        number_of_clusters = 4

        # Loop over files and get features
        file_list = glob.glob((dataset_dir_path / '*.jpg').as_posix())
        # file_list = ["1.jpg", "2.jpg", "3.jpg", .... ]
        feature_list = []
        # feature_list = [[feat_1], [feat_2], [feat_3], ... ]
        for image_path in tqdm(file_list):
            try:
                img = image.load_img(image_path, target_size=(224, 224))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                features = np.array(model.predict(img_data))
                feature_list.append(features.flatten())
            except Exception as e:
                continue

        # Clustering
        kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(np.array(feature_list))
        for i in range(len(file_list)):
            file_path = file_list[i]
            cluster_idx = kmeans.labels_[i]

            cluster_idx_dir = results_dir_path / str(cluster_idx)
            if not cluster_idx_dir.exists():
                cluster_idx_dir.mkdir()

            shutil.copy(file_path, cluster_idx_dir.as_posix())

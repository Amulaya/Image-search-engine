from keras.preprocessing import image
from keras.applications.vgg16 import VGG16  # a pretrained model used to just identify the images
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import shutil, glob, os.path


# Shutil module in Python provides many functions of high-level operations on files and collections of files.
# It comes under Python's standard utility modules.
# This module helps in automating process of copying and removal of files and directories.
class Utils:
    @classmethod
    def clustring_of_images(self, path_of_project_dir: str):
        image.LOAD_TRUNCATED_IMAGES = True
        model = VGG16(weights='imagenet', include_top=False)

        # Variables
        # DIR containing images
        path_of_dir_clusterd_images = path_of_project_dir + "/sorted_images/"
        # DIR to copy clustered images to new DIR

        number_of_clusters = 4

        # Loop over files and get features
        file_list = glob.glob(os.path.join(path_of_project_dir+"/dataset/", '*.jpg'))
        file_list.sort()
        feature_list = []
        for i, imagepath in enumerate(file_list):
            try:
                print("    Status: %s / %s" % (i + 1, len(file_list)), end="\r")
                img = image.load_img(imagepath, target_size=(224, 224))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                features = np.array(model.predict(img_data))
                feature_list.append(features.flatten())
            except:
                continue

        # Clustering
        kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(np.array(feature_list))
        # Copy images renamed by cluster
        # Check if target dir exists
        try:
            os.makedirs(path_of_dir_clusterd_images)
        except OSError:
            print("OSError: The directory already exists please delete and again run it")
        # Copy with cluster name
        print("\n")
        for i, m in enumerate(kmeans.labels_):
            try:
                print("Copy: %s / %s" % (i, len(kmeans.labels_)), end="\r")
                shutil.copy(file_list[i], path_of_dir_clusterd_images + str(m) + "_" + str(i) + ".jpg")
            except:
                continue

    @classmethod
    def get_clustred(self, path_of_project_dir):
        new_dic = path_of_project_dir + "/cluster_of_sorted_images/"
        os.mkdir(new_dic)
        new_path = path_of_project_dir + "/cluster_of_sorted_images/"
        path = path_of_project_dir + "/sorted_images/"
        for i in os.listdir(path):
            folder_name = i.split('_')
            if folder_name[0] not in os.listdir(new_dic):
                folder_dir = (new_path + folder_name[0])
                os.mkdir(folder_dir)
            else:
                shutil.copy(path_of_project_dir + "/sorted_images/" + i,
                            new_path + folder_name[0] + "/" + folder_name[1])

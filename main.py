import matplotlib.pyplot as plt
from src.img_cluster import ImgCluster
from src.utils import Utils
import cv2
from tqdm import tqdm

def main():
    progess = tqdm()
    progess.update()
    print("Enter the directory having this project")
    path_of_project_dir = input()
    print("Enter the image number whose results you want to see")
    image_no_for_query = input()
    path_of_query = path_of_project_dir + "/dataset/" + image_no_for_query + ".jpg"
    print("Enter the no of simliar images you want to see")
    N = int(input())
    Utils.clustring_of_images(path_of_project_dir)
    Utils.get_clustred(path_of_project_dir)
    print("The clusters have been formed correctly in the directory name cluster_of_sorted_images!!")

    ImgCluster.index(path_of_project_dir)
    print("The index.csv file containing the image and its color descriptor has created!!")
    list_of_labels_for_animals = ImgCluster.get_closest_match(path_of_project_dir,path_of_query,N)
    print("The required N images have been identified and will be shown in the plot")
    query = cv2.imread(path_of_query)
    query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(5, 5, 1)
    plt.imshow(query)
    for i in range(1, len(list_of_labels_for_animals) + 1):
        img = plt.imread(path_of_project_dir+'/dataset/' + list_of_labels_for_animals[i - 1])
        fig.add_subplot(5, 5, i + 5)
        plt.imshow(img)
    print("Here is your plot showing N images acc to query :)")
    plt.title("Images similar to input images")
    plt.show()


if __name__ == "__main__":
    main()

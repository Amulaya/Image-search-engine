import glob
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from src.img_cluster_1 import Searcher

from src.img_cluster_0 import ColorDescriptor


class ImgCluster:
    def get_closet_images(self, dataset_dir_path: Path, results_dir_path: Path, image_no_for_query: str, N: int):
        cd = ColorDescriptor((8, 12, 3))
        # open the output index file for writing
        output = Path.open(results_dir_path / "index.csv", "w")
        # use glob to grab the image paths and loop over them
        print("Generating the index file with images and extracted features")
        for imagePath in tqdm(glob.glob((dataset_dir_path / "*.jpg").as_posix())):
            # extract the image ID (i.e. the unique filename) from the image
            # path and load the image itself
            imageID = imagePath[imagePath.rfind("/") + 1:]
            image = cv2.imread(imagePath)
            # describe the image
            features = cd.describe(image)
            # write the features to file
            features = [str(f) for f in features]
            output.write("%s,%s\n" % (imageID, ",".join(features)))
        # close the index file
        output.close()

        query_dir_path = Path(dataset_dir_path / Path(str(image_no_for_query) + (".jpg")))
        cd = ColorDescriptor((8, 12, 3))
        # load the query image and describe it
        query = cv2.imread(query_dir_path.as_posix())
        features = cd.describe(query)
        # perform the search
        searcher = Searcher(results_dir_path / "index.csv")
        results = searcher.search(features, N)
        # display the query
        # loop over the results
        print("Generating the results !!")
        final = []
        for (_, resultID) in results:
            final.append(resultID)

        query = cv2.imread(query_dir_path.as_posix())
        query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(5, 5, 1)
        plt.imshow(query)
        print("Plotting the results obtained")
        for i in tqdm(range(1, len(final) + 1)):
            img = plt.imread(dataset_dir_path / final[i - 1])
            fig.add_subplot(5, 5, i + 5)
            plt.imshow(img)
        plt.show()

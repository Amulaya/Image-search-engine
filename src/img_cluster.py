import glob
import cv2

from src.chi2_distance_based_search import Searcher

from src.color_descriptor_of_images import ColorDescriptor


class ImgCluster:
    @classmethod
    def index(self, path_of_project_dir: str):
        cd = ColorDescriptor((8, 12, 3))
        # open the output index file for writing
        output = open(path_of_project_dir + "/index.csv", "w")
        # use glob to grab the image paths and loop over them
        for imagePath in glob.glob(path_of_project_dir + "/dataset" + "/*.jpg"):
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

    @classmethod
    def get_closest_match(self, path_of_project_dir, path_of_query,N):
        cd = ColorDescriptor((8, 12, 3))
        # load the query image and describe it
        query = cv2.imread(path_of_query)
        features = cd.describe(query)
        # perform the search
        searcher = Searcher(path_of_project_dir + "/index.csv")
        results = searcher.search(features, N)
        # display the query
        # loop over the results
        final = []
        for (_, resultID) in results:
            final.append(resultID)
        return final

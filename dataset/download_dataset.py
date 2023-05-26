import os


def download_dataset(dataset=None, force=False, unzip=True):
    """Download a dataset from Kaggle.

    Args:
        dataset (str): The dataset to download with the format <owner>/<dataset-name>. If None, download all datasets in dataset/dataset.txt. (Default: None)
        force (bool): Force download the dataset. (Default: False)
        unzip (bool): Unzip the dataset and remove the zip file. (Default: True)
    """
    DATASET_DIR = os.path.join(os.path.dirname(__file__))

    if force:
        force = "--force"
    else:
        force = ""

    if unzip:
        unzip = "--unzip"
    else:
        unzip = ""

    if dataset is None:
        with open(os.path.join(DATASET_DIR, "dataset.txt"), "r") as f:
            for dataset in f:
                filename = dataset.split("/")[-1].split(".")[0].strip()
                dataset_path = os.path.join(DATASET_DIR, filename)
                os.system(
                    "kaggle datasets download -d {} {} {} -p {}".format(
                        dataset, force, unzip, dataset_path
                    )
                )
                print("Downloaded {}".format(dataset))
    else:
        filename = dataset.split("/")[-1].split(".")[0].strip()
        dataset_path = os.path.join(DATASET_DIR, filename)
        os.system(
            "kaggle datasets download -d {} {} {} -p {}".format(
                dataset, force, unzip, dataset_path
            )
        )
        print("Downloaded {}".format(dataset))


if __name__ == "__main__":
    download_dataset(force=True, unzip=True)

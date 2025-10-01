#NOTE - the dataset can be downloaded onto your drive from: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
#Copying the zipped dataset into the local directory of Google Colab/Locally.
#NOTE - Change the dataset path and destination path with your own file path.
import zipfile
zip_ref = zipfile.ZipFile("/content/drive/MyDrive/ML/BraTS2020.zip", 'r')
zip_ref.extractall("/content/drive/MyDrive/ML/BraTS2020")
zip_ref.close()

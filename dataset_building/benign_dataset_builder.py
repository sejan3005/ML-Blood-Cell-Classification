import glob
import numpy as np
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt

folder_path = './Segmented/Benign'

benign_images = glob.glob(os.path.join(folder_path,"**","*.jpg"),recursive=True)

# feature engineering every image from the glob
image_array = []

for image in benign_images:
    
    # open the image
    img = Image.open(image)
    
    # convert the image
    img = img.convert("L")
    
    # overwrite the image with its pixels
    img = img.getdata()
    
    # overwrite it again with an array form
    img = np.array(img)

    # over write again with normal values
    img = img/255

    image_array.append(img)

columns = [f'pixel{i+1}' for i in range(50176)]

benign_frame = pd.DataFrame(image_array,columns=columns)

benign_frame["diagnosis"] = "benign"


print(benign_frame.head())

# plot one of the images for fun


def plot_image(instance,df=benign_frame):
    """ in this function you enter the instance of the benign data frame that you want plotted """
    
    df = df.drop(columns="diagnosis")

    df = np.array(df.loc[instance]).reshape(224,224)

    plt.imshow(df,cmap="gray")
    plt.show()


plot_image(7)

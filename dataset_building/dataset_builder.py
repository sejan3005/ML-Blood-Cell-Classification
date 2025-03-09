import glob
import numpy as np
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt

# first create the DataFrame where the benign images will be stored

folder_path_benign = './Segmented/Benign'

benign_images = glob.glob(os.path.join(folder_path_benign,"**","*.jpg"),recursive=True)

# feature engineering every image from the  benign glob
benign_image_array = []

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

    benign_image_array.append(img)

benign_columns = [f'pixel{i+1}' for i in range(50176)]

benign_frame = pd.DataFrame(benign_image_array,columns=benign_columns)

benign_frame["diagnosis"] = "benign"


print(benign_frame.head())

# plot one of the images for fun





# plot_image(7)


# the following code builds the early diagnosis dataframe

folder_path_early = "./Segmented/Early"

early_images = glob.glob(os.path.join(folder_path_early,"**","*.jpg"),recursive=True)

early_image_array = []

for image in early_images:

    # open the image
    img = Image.open(image)

    # convert the image to grayscale
    img = img.convert("L")

    # overwrite with pixels
    img =img.getdata()

    # overwrite again with an array
    img = np.array(img)

    # overwrite with normal values

    img = img/255

    early_image_array.append(img)


early_columns = [f'pixel{i+1}' for i in range(50176)]

early_frame = pd.DataFrame(early_image_array,columns=early_columns)

early_frame["diagnosis"] = "early"

print(early_frame.head())

# the following code will build the pre diagnosis dataframe

folder_path_pre = "./Segmented/Pre"

pre_images = glob.glob(os.path.join(folder_path_pre,"**","*.jpg"),recursive=True)

pre_image_array = []

for image in pre_images:

    # open the image
    img = Image.open(image)

    # convert to grayscale
    img = img.convert("L")

    # overwrite with pixels
    img = img.getdata()

    # overwrite with an array
    img = np.array(img)

    # over write with normal values
    img = img/255

    pre_image_array.append(img)


pre_columns = [f"pixel{i+1}" for i in range(50176)]

pre_frame  = pd.DataFrame(pre_image_array,columns = pre_columns)

pre_frame["diagnosis"] = "pre"

print(pre_frame.head())

# the following code will build the pro diagnosis dataframe

folder_path_pro = "./Segmented/Pro"

pro_images = glob.glob(os.path.join(folder_path_pro,"**","*.jpg"),recursive=True)

pro_image_array = []

for image in pro_images:

    # open the image
    img = Image.open(image)

    # convert to gray scale
    img = img.convert("L")

    # overwrite with pixels
    img = img.getdata()

    # overwrite with an array
    img = np.array(img)

    # overwrite with normal
    img = img/255

    pro_image_array.append(img)

pro_columns = [f"pixel{i+1}" for i in range(50176)]

pro_frame = pd.DataFrame(pro_image_array,columns = pro_columns)

pro_frame["diagnosis"] = "pro"

print(pro_frame.head())


def plot_image(instance,df=benign_frame):
    """ in this function you enter the instance of the benign data frame that you want plotted """
    
    df = df.drop(columns="diagnosis")

    df = np.array(df.loc[instance]).reshape(224,224)

    plt.imshow(df,cmap="gray")
    plt.show()




# now we need to concat every single data frame we built into one DataFrame and then export it as a csv file

leukemia_frame = pd.concat([benign_frame,early_frame,pre_frame,pro_frame],axis=0)

leukemia_frame.to_csv("leukemia_frame.csv",index=False)
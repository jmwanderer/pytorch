"""
Illustrations of convolutional image processing

"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import images



def to_image(data):
    width = data.shape[0] * data.shape[2]
    height = data.shape[1]
    a = np.zeros((height, width))
    for i in range(data.shape[0]):
        for x in range(0, data.shape[2]):
            for y in range(0, height):
                a[y][x+i*data.shape[2]] = int(255 * data[i][y][x])
    return a

def get_images(data):
    images = []
    images.append(to_image(data))

    data = model.conv1(data)
    data = model.relu(data)
    images.append(to_image(data))
    
    data = model.maxpool(data)
    images.append(to_image(data))

    data = model.conv2(data)
    data = model.relu(data)
    images.append(to_image(data))

    data = model.maxpool(data)
    images.append(to_image(data))
    return images

model = images.load_model(use_faces_data=True, use_conv_model=True)
training_data, test_data, image_size, label_classes, batch_size, learning_rate = images.get_training_data(use_faces_data=True, use_conv_model=True)

if len(sys.argv) > 0 and sys.argv[0] == 'detail':
    detail = True
else:
    detail = False

if not detail:
    f, axs = plt.subplots(10,2)
    for index in range(len(training_data)):
        print(f"read index {index}")
        data, label = training_data[index]
        images = get_images(data)

        axs[index % 10][0].imshow(images[0])    
        axs[index % 10][1].imshow(images[4])    

        if index % 10 == 9:
            plt.show()
            f, axs = plt.subplots(10,2)
else:
    for index in range(len(training_data)):
        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
        print(f"read index {index}")
        data, label = training_data[index]
        images = get_images(data)

        ax1.imshow(images[0])
        ax2.imshow(images[1])
        ax3.imshow(images[2])
        ax4.imshow(images[3])
        ax5.imshow(images[4])
        plt.show()

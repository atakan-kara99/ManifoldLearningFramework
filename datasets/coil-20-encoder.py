import numpy as np
from PIL import Image
import os
from tempfile import TemporaryFile


coil20_path = 'datasets//coil-20-proc'

#init_labels = [10,11,12,13,14,15,16,17,18,19,1,20,2,3,4,5,6,7,8,9]
# init_labels = list(range(21,41))
new_labels = list(range(1,21))
init_image_nums = list(range(72))

images = []
labels = []
image_nums = []



# for i, label in enumerate(new_labels):
#     for image_num in init_image_nums:
#         img_path = 'obj' + str(label) + '__' + str(image_num) + '.png'
#         #new_img_path = 'obj' + str(new_labels[i]) + '__' + str(image_num) + '.png'
#         #new_image_path = os.path.join(coil20_path, new_img_path)
#         #os.rename(image_path, new_image_path)
#         image_path = os.path.join(coil20_path, img_path)
#         image = Image.open(image_path)
#         images.append(np.array(image).flatten())
#         labels.append(label)
#         image_nums.append(image_num)

# with open('datasets//coil-20-encoded.npy', 'wb') as f:
#     np.save(f, np.array(images))
#     np.save(f, np.array(labels))
#     np.save(f, np.array(image_nums))

with open('datasets//coil-20-encoded.npy', 'rb') as f:
    images = np.load(f)
    labels = np.load(f)
    image_nums = np.load(f)

# print(images)
# curr = -1
# for i in labels:
#     if i != curr:
#         print(i)
#     curr = i
# print(labels[1367])
# for i in range(72):
#     print(image_nums[i])
print(labels)
print(image_nums)
print('obj' + str(labels[1367]) + '__' + str(image_nums[1367]) + '.png')
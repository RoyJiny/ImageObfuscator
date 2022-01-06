import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import random
import sys

SIZE = 512

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print(f'Usage: {sys.argv[0]} first_image second_image secret_image output_image1 output_image2')
        exit(1)

    # import images and convert to black and white
    imageA_grayscale = color.rgb2gray(plt.imread(sys.argv[1]))[:SIZE,:SIZE]
    imageA_grayscale = np.array(imageA_grayscale).flatten()
    imageA_th = np.max(imageA_grayscale)/2
    imageA = [int(p > imageA_th) for p in imageA_grayscale]

    imageB_grayscale = color.rgb2gray(plt.imread(sys.argv[2]))[:SIZE,:SIZE]
    imageB_grayscale = np.array(imageB_grayscale).flatten()
    imageB_th = np.max(imageB_grayscale)/2
    imageB = [int(p > imageB_th) for p in imageB_grayscale]

    secret_img_grayscale = color.rgb2gray(plt.imread(sys.argv[3]))[:SIZE,:SIZE]
    secret_img_grayscale = np.array(secret_img_grayscale).flatten()
    secret_img_th = np.max(secret_img_grayscale)/2
    secret_img = [int(p > secret_img_th) for p in secret_img_grayscale]


    # For white C pixel:
    def create_block_pair_for_white_C_value(A_pixel,B_pixel):
        A_block = np.zeros(4)
        B_block = np.zeros(4)
        indices = [i for i in range(4)]
        if A_pixel == 1 and B_pixel == 1:
            A_indices = random.sample(indices,2) # 2 white
            B_indices = random.sample(A_indices,1) # 1 to overlap
            B_indices += random.sample(set(indices)-set(A_indices),1) # 1 that doesnt overlap
        if A_pixel == 1 and B_pixel == 0:
            A_indices = random.sample(indices,2) # 2 white
            B_indices = random.sample(A_indices,1) # 1 that overlaps
        if A_pixel == 0 and B_pixel == 1:
            B_indices = random.sample(indices,2) # 2 white
            A_indices = random.sample(B_indices,1) # 1 that overlaps
        if A_pixel == 0 and B_pixel == 0:
            A_indices = random.sample(indices,1) # 1 white
            B_indices = A_indices # same one for overlapping

        np.put(A_block,A_indices,1)
        np.put(B_block,B_indices,1)
        return (np.reshape(A_block,(2,2)),np.reshape(B_block,(2,2)))

    # For black C pixel:
    def create_block_pair_for_black_C_value(A_pixel,B_pixel):
        A_block = np.zeros(4)
        B_block = np.zeros(4)
        indices = [i for i in range(4)]
        if A_pixel == 1 and B_pixel == 1:
            A_indices = random.sample(indices,2) # 2 white
            B_indices = [i for i in set(indices)-set(A_indices)] # take the other 2
        if A_pixel == 1 and B_pixel == 0:
            A_indices = random.sample(indices,2) # 2 white
            B_indices = random.sample(set(indices)-set(A_indices),1) # 1 that doesnt overlap
        if A_pixel == 0 and B_pixel == 1:
            B_indices = random.sample(indices,2) # 2 white
            A_indices = random.sample(set(indices)-set(B_indices),1) # 1 that doesnt overlap
        if A_pixel == 0 and B_pixel == 0:
            A_indices = random.sample(indices,1) # 1 white
            B_indices = random.sample(set(indices)-set(A_indices),1) # 1 that doesnt overlap

        np.put(A_block,A_indices,1)
        np.put(B_block,B_indices,1)
        return (np.reshape(A_block,(2,2)),np.reshape(B_block,(2,2)))


    def construct_image_from_blocks(block_list):
        img = [None for _ in range((SIZE*2)*(SIZE*2))]
        for idx,block in enumerate(block_list):
            row = idx // SIZE
            i = (idx%SIZE)*2 + row*(SIZE*2)*2
            img[i] = block[0][0]
            img[i+1] = block[0][1]
            img[i+(SIZE*2)] = block[1][0]
            img[i+SIZE*2+1] = block[1][1]
        return np.array(img).reshape(((SIZE*2),(SIZE*2)))

    def place_images(im1,im2):
        flat_im1 = im1.flatten()
        flat_im2 = im2.flatten()
        flat = flat_im1+flat_im2
        flat = (flat == 2).astype('int')
        return np.reshape(flat,((SIZE*2),(SIZE*2)))

    A_img_blocks = []
    B_img_blocks = []
    for idx,pixel in enumerate(secret_img):
        if pixel == 1:
            A_block,B_block = create_block_pair_for_white_C_value(imageA[idx],imageB[idx])
        else:
            A_block,B_block = create_block_pair_for_black_C_value(imageA[idx],imageB[idx])
        A_img_blocks.append(A_block)
        B_img_blocks.append(B_block)

    A_img = construct_image_from_blocks(A_img_blocks)
    B_img = construct_image_from_blocks(B_img_blocks)


    fig, ax = plt.subplots(3, 2, figsize=(10,10))
    fig.tight_layout()

    ax[0,0].imshow(np.reshape(imageA,(SIZE,SIZE)), cmap='gray', vmax=1,vmin=0)
    ax[0,0].title.set_text("image A")

    ax[0,1].imshow(A_img, cmap='gray', vmax=1,vmin=0)
    ax[0,1].title.set_text("image A~")

    ax[1,0].imshow(np.reshape(imageB,(SIZE,SIZE)), cmap='gray', vmax=1,vmin=0)
    ax[1,0].title.set_text("image B")

    ax[1,1].imshow(B_img, cmap='gray', vmax=1,vmin=0)
    ax[1,1].title.set_text("image B~")

    ax[2,0].imshow(np.reshape(secret_img,(SIZE,SIZE)), cmap='gray', vmax=1,vmin=0)
    ax[2,0].title.set_text("image C")

    ax[2,1].imshow(place_images(A_img,B_img), cmap='gray', vmax=1,vmin=0)
    ax[2,1].title.set_text("image C~")

    plt.show()

    plt.imsave(sys.argv[4],A_img, cmap='gray', vmax=1, vmin=0)
    plt.imsave(sys.argv[5],B_img, cmap='gray', vmax=1, vmin=0)


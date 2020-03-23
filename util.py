import hashlib
import math
import matplotlib.pyplot as plt
import cv2
import os

def show_images(images, ncols = 1):
    nrows =  math.ceil(len(images) / ncols)
    fig=plt.figure(figsize=(10,5))
    for ind,img in enumerate(images):
        ax = fig.add_subplot(nrows, ncols, ind+1)
        ax.imshow(img)
        ax.grid('on', linestyle='--')
        ax.axis('off')
        ax.set_title("{},{}".format(img.shape[0],img.shape[1]))
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()

def conv_output_shape(h_w,kernel_size=1,stride=1,pad=0,dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(h_w) is not tuple:
        h_w=(h_w,h_w)

    if type(kernel_size) is not tuple:
        kernel_size=(kernel_size,kernel_size)

    if type(stride) is not tuple:
        stride=(stride,stride)

    if type(pad) is not tuple:
        pad=(pad,pad)

    h=(h_w[0]+(2*pad[0])-(dilation*(kernel_size[0]-1))-1)//stride[0]+1
    w=(h_w[1]+(2*pad[1])-(dilation*(kernel_size[1]-1))-1)//stride[1]+1
    return h,w


def convtransp_output_shape(h_w,kernel_size=1,stride=1,pad=0,dilation=1):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(h_w) is not tuple:
        h_w=(h_w,h_w)

    if type(kernel_size) is not tuple:
        kernel_size=(kernel_size,kernel_size)

    if type(stride) is not tuple:
        stride=(stride,stride)

    if type(pad) is not tuple:
        pad=(pad,pad)

    h=(h_w[0]-1)*stride[0]-2*pad[0]+kernel_size[0]+pad[0]
    w=(h_w[1]-1)*stride[1]-2*pad[1]+kernel_size[1]+pad[1]

    return h,w


def generate_sha256_unique_file_name(file):
    assert os.path.isfile(file), "Invalid file name"
    file_fullname = os.path.basename(file)
    file_name, file_ext  = os.path.splitext(file_fullname)
    with open(file, 'rb') as f:
        hash_object=hashlib.sha256(f.read())
        hex_dig=hash_object.hexdigest()
        return "{}-{}{}".format(file_name, hex_dig[:8], file_ext)

if __name__ == '__main__':
    file_name = generate_sha256_unique_file_name("convmodel.pt")
    print(file_name)
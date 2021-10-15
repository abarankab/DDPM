import argparse
import matplotlib.pyplot as plt
import torchvision

from matplotlib import animation
from IPython.display import HTML, display


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def get_transform():
    class RescaleChannels(object):
        def __call__(self, sample):
            return 2 * sample - 1

    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RescaleChannels(),
    ])


def cycle(dl):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for data in dl:
            yield data


def preprocess_image(img):
    plt.axis('off')

    if img.shape[0] == 3:
        return plt.imshow(((img + 1) / 2).permute(1, 2, 0), animated=True)
    elif img.shape[0] == 1:
        return plt.imshow(((img + 1) / 2).squeeze(), animated=True, cmap="gray")
    else:
        raise ValueError("incorrect number of channels in an image")


def show_video(samples_list, delay=100, id=0):
    fig = plt.figure(figsize=(8, 8))
    images = []
    for sample in samples_list:
        images.append([preprocess_image(sample[id])])
    
    ani = animation.ArtistAnimation(fig, images, interval=delay)
    display(HTML(ani.to_html5_video()))


def show_images(image_tensor, rows, cols=None, colorbar=False):
    if cols is None:
        cols = (len(image_tensor) + rows - 1) // rows

    if image_tensor[0].shape[0] == 3:
        is_rgb = True
        if colorbar:
            raise ValueError("cannot display colorbar with rgb images")
    elif image_tensor[0].shape[0] == 1:
        is_rgb = False
    else:
        raise ValueError("incorrect number of channels in an image")

    plt.figure(figsize=(4 * cols, 4 * rows))

    for i in range(rows):
        for j in range(cols):
            if i * cols + j == len(image_tensor):
                break

            plt.subplot(rows, cols, i * cols + j + 1)
            if is_rgb:
                plt.imshow(((image_tensor[i * cols + j] + 1) / 2).permute(1, 2, 0))
            else:
                plt.imshow(((image_tensor[i * cols + j] + 1) / 2).squeeze(), cmap="gray")
            
            if colorbar:
                plt.colorbar()
    
    plt.show()


def show_image(img, colorbar=False):
    if img.shape[0] == 3:
        if colorbar:
            raise ValueError("cannot display colorbar with rgb images")
        plt.imshow(((img + 1) / 2).permute(1, 2, 0), animated=True)
    elif img.shape[0] == 1:
        plt.imshow(((img + 1) / 2).squeeze(), animated=True, cmap="gray")
        if colorbar:
            plt.colorbar()
    else:
        raise ValueError("incorrect number of channels in an image")
    
    plt.show()


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

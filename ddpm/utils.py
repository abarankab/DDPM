import matplotlib.pyplot as plt

from matplotlib import animation
from IPython.display import HTML, display


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


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

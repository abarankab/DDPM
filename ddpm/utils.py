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


def generate_video(samples_list, delay=100, id=0):
    fig = plt.figure(figsize=(8, 8))
    images = []
    for sample in samples_list:
        images.append([preprocess_image(sample[id].squeeze())])
    
    ani = animation.ArtistAnimation(fig, images, interval=delay)
    display(HTML(ani.to_html5_video()))

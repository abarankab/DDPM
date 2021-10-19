import argparse
import torch
import torchvision

from ddpm import script_utils


def main():
    args = create_argparser().parse_args()
    device = args.device

    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        diffusion.load_state_dict(torch.load(args.model_path))

        if args.use_labels:
            for label in range(10):
                y = torch.ones(args.num_images // 10, dtype=torch.long, device=device) * label
                samples = diffusion.sample(args.num_images // 10, device, y=y)

                for image_id in range(len(samples)):
                    image = ((samples[image_id] + 1) / 2).clip(0, 1)
                    torchvision.utils.save_image(image, f"{args.save_dir}/{label}-{image_id}.png")
        else:
            samples = diffusion.sample(args.num_images, device)

            for image_id in range(len(samples)):
                image = ((samples[image_id] + 1) / 2).clip(0, 1)
                torchvision.utils.save_image(image, f"{args.save_dir}/{image_id}.png")
    except KeyboardInterrupt:
        print("Keyboard interrupt, generation finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(num_images=10000, device=device)
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str)
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
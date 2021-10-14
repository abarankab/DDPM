import argparse
import torch
import torch.nn.functional as F
import wandb

from torch.utils.data import DataLoader
from torchvision import datasets
from . import utils
from .unet import UNet
from .diffusion import GaussianDiffusion, generate_linear_schedule, generate_cosine_schedule


def create_argparser():
    defaults = dict(
        num_timesteps=1000,
        schedule="linear",
        loss_type="l2",
        use_labels=False,

        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        time_emb_dim=10,
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(1,),

        learning_rate=2e-4,
        batch_size=128,
        iterations=800000,
        
        ema_decay=0.9999,
        ema_update_rate=1,

        log_to_wandb=True,
        log_rate=1000,
        log_dir="~/ddpm_logs"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str)
    utils.add_dict_to_argparser(parser, defaults)
    return parser


def train():
    args = create_argparser().parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    try:
        activations = {
            "relu": F.relu,
            "mish": F.mish,
            "silu": F.silu,
        }

        model = UNet(
            img_channels=3,

            base_channels=args.base_channels,
            channel_mults=args.channel_mults,
            time_emb_dim=args.time_emb_dim,
            norm=args.norm,
            dropout=args.dropout,
            activation=activations[args.activation],
            attention_resolutions=args.attention_resolutions,

            num_classes=None if not args.use_labels else 10,
            initial_pad=0,
        )

        if args.schedule == "cosine":
            betas = generate_cosine_schedule(args.num_timesteps)
        else:
            betas = generate_linear_schedule(
                args.num_timesteps,
                1e-4 * 1000 / args.num_timesteps,
                0.02 * 1000 / args.num_timesteps,
            )

        diffusion = GaussianDiffusion(
            model, (32, 32), 3, 10,
            betas,
            ema_decay=args.ema_decay,
            ema_update_rate=args.ema_update_rate,
            ema_start=2000,
            loss_type=args.loss_type,
        ).to(device)
        
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

        if args.log_to_wandb:
            run = wandb.init(
                project=args.project_name,
                entity='treaptofun',
                config=vars(args),
            )

            run.name = run.id
            wandb.watch(diffusion)

        batch_size = args.batch_size

        train_dataset = datasets.CIFAR10(
            root='./cifar_train',
            train=True,
            download=True,
            transform=utils.get_transform(),
        )

        test_dataset = datasets.CIFAR10(
            root='./cifar_test',
            train=False,
            download=True,
            transform=utils.get_transform(),
        )

        train_loader = utils.cycle(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=4)
        
        acc_train_loss = 0

        for iteration in range(1, args.iterations + 1):
            diffusion.train()

            x, y = next(train_loader)
            x = x.to(device)
            y = y.to(device)

            if args.use_labels:
                loss = diffusion(x, y)
            else:
                loss = diffusion(x)

            acc_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            diffusion.update_ema()
            
            if iteration % args.log_rate == 0:
                test_loss = 0
                with torch.no_grad():
                    diffusion.eval()
                    for x, y in test_loader:
                        x = x.to(device)
                        y = y.to(device)

                        if args.use_labels:
                            loss = diffusion(x, y)
                        else:
                            loss = diffusion(x)

                        test_loss += loss.item()

                model_filename = f"{args.log_dir}/{args.project_name}-{run.name}-{iteration}-model.pth"
                optim_filename = f"{args.log_dir}/{args.project_name}-{run.name}-{iteration}-optim.pth"

                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)
                
                if args.use_labels:
                    samples = diffusion.sample(10, device, y=torch.arange(10, device=device))
                else:
                    samples = diffusion.sample(10, device)
                
                samples = ((samples + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy()

                test_loss /= args.log_rate
                acc_train_loss /= args.log_rate

                wandb.log({
                    "test_loss": test_loss,
                    "train_loss": acc_train_loss,
                    "samples": [wandb.Image(sample) for sample in samples],
                })

                acc_train_loss = 0
                
        run.finish()
    except KeyboardInterrupt:
        run.finish()
        print("Keyboard interrupt, run finished early")

import argparse
import torch
import torch.nn.functional as F
import utils
import wandb

from unet import UNet
from diffusion import GaussianDiffusion, generate_linear_schedule, generate_cosine_schedule


def create_argparser():
    defaults = dict(
        num_timesteps=1000,
        schedule="linear",
        loss_type="l2",

        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        time_emb_dim=10,
        norm="gn",
        dropout=0.1,
        activation=F.silu,
        attention_resolutions=(1,),

        learning_rate=2e-4,
        batch_size=128,
        iterations=800000,
        
        ema_decay=0.9999,
        ema_update_rate=1,

        use_wandb=True,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str)
    utils.add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    try:
        model = UNet(
            img_channels=3,

            base_channels=args.base_channels,
            channel_mults=args.channel_mults,
            time_emb_dim=args.time_emb_dim,
            norm=args.norm,
            dropout=args.dropout,
            activation=args.activation,
            attention_resolutions=args.attention_resolutions,

            num_classes=None,
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

        if use_wandb:
            run = wandb.init(project='ddpm-debug', entity='treaptofun', reinit=True, cnonfig=vars(args))
            print(run.name)
            wandb.watch(diffusion)

        batch_size = args.batch_size

        train_dataset = datasets.CIFAR10(root='./cifar_train', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./cifar_test', train=False, download=True, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=4)

        epochs = config["epochs"]
        grad_losses_train = []
        grad_losses_test = []

        mb = range(config["epochs"])

        for epoch in mb:
            epoch_loss = 0
            diffusion.train()
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                loss = diffusion(x, y)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                diffusion.update_ema()

            grad_losses_train.append(epoch_loss / len(train_loader))

            epoch_loss = 0
            diffusion.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)

                    loss = diffusion(x, y)
                    epoch_loss += loss.item()

            grad_losses_test.append(epoch_loss / len(test_loader))

            print(epoch)

            if (epoch + 1) % 100 == 0:
                model_filename = f"{run.name}-{epoch}.pth"
                optim_filename = f"{run.name}-optim-{epoch}.pth"
                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)
                print(f"Saved checkpoint at epoch {epoch} under name {model_filename}")
                
            samples = ((diffusion.sample(10, device, y=torch.arange(10, device=device)) + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy()

            wandb.log({
                "test_loss": grad_losses_test[-1],
                "train_loss": grad_losses_train[-1],
                "samples": [wandb.Image(sample) for sample in samples],
            })
                
        run.finish()
    except KeyboardInterrupt:
        run.finish()
        print("Keyboard interrupt, run finished early")
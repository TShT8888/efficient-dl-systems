import argparse
import torch
from torch import nn
from tqdm.auto import tqdm

from unet import Unet
from dataset import get_train_data


def unscale_grads_(optimizer: torch.optim.Optimizer, scale: float) -> None:
    """Divide gradients by scale in-place."""
    inv_scale = 1.0 / float(scale)
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                p.grad.data.mul_(inv_scale)


@torch.no_grad()
def has_inf_or_nan_grads(optimizer: torch.optim.Optimizer) -> bool:
    """True if any grad has inf or nan."""
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad.data
            if not torch.isfinite(g).all():
                return True
    return False


class DynamicLossScaler:
    """
    Simple dynamic loss scaler:
      - if overflow: scale *= backoff_factor, skip optimizer.step()
      - if growth_interval good steps in a row: scale *= growth_factor
    """

    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 200,
        min_scale: float = 1.0,
        max_scale: float = 2**24,
    ):
        self.scale = float(init_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self._good_steps = 0

    def update(self, overflow: bool) -> None:
        if overflow:
            self.scale = max(self.min_scale, self.scale * self.backoff_factor)
            self._good_steps = 0
        else:
            self._good_steps += 1
            if self._good_steps >= self.growth_interval:
                self.scale = min(self.max_scale, self.scale * self.growth_factor)
                self._good_steps = 0


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode: str = "dynamic",  # "static" | "dynamic"
    dyn_scaler: DynamicLossScaler | None = None,
    static_scale: float = 1024.0,
) -> None:
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for i, (images, labels) in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # forward in AMP
        with torch.amp.autocast(device.type, dtype=torch.float16):
            outputs = model(images)        
            loss = criterion(outputs, labels)


        if mode == "static":
            scale = static_scale
            (loss * scale).backward()
            unscale_grads_(optimizer, scale)
            optimizer.step()

        elif mode == "dynamic":
            if dyn_scaler is None:
                raise ValueError("dyn_scaler must be provided for mode='dynamic'")

            scale = dyn_scaler.scale
            (loss * scale).backward()

            unscale_grads_(optimizer, scale)
            overflow = has_inf_or_nan_grads(optimizer)

            if not overflow:
                optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)

            dyn_scaler.update(overflow)

        else:
            raise ValueError("mode must be 'static' or 'dynamic'")

        # metrics (BCEWithLogitsLoss -> sigmoid first)
        probs = torch.sigmoid(outputs)
        accuracy = ((probs > 0.5) == labels).float().mean()

        extra = ""
        if mode == "dynamic":
            extra = f" scale={dyn_scaler.scale:.1f} overflow={int(overflow)}"
        else:
            extra = f" scale={static_scale:.1f}"

        pbar.set_description(f"Loss: {loss.item():.4f} Acc: {accuracy.item() * 100:.2f}%{extra}")


def train(mode: str = "dynamic") -> None:
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()

    num_epochs = 5

    dyn_scaler = None
    if mode == "dynamic":
        dyn_scaler = DynamicLossScaler(init_scale=65536.0, growth_interval=200)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs} | mode={mode}")
        train_epoch(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            mode=mode,
            dyn_scaler=dyn_scaler,
            static_scale=1024.0,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="dynamic", choices=["static", "dynamic"])
    args = parser.parse_args()
    train(mode=args.mode)


if __name__ == "__main__":
    main()
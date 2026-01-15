#!/usr/bin/env python3
"""
Training script for beam propagation CNN.

Architecture:
    Input: width × 2 channels (beams + grid)
    Conv1D(kernel=3, in=2, out=8) + ReLU
    Conv1D(kernel=3, in=8, out=8) + ReLU
    Conv1D(kernel=3, in=8, out=1) + Sigmoid
    Output: width × 1 (next row beams)

The network predicts the next row's beam state given the current state
and the splitter positions in the current row.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


class BeamDataset(Dataset):
    """Dataset for beam propagation training."""

    def __init__(self, csv_path: str, width: int = 16):
        """Load training data from CSV.

        CSV format: beam_0,...,beam_w-1,grid_0,...,grid_w-1,next_0,...,next_w-1
        """
        df = pd.read_csv(csv_path)
        self.width = width

        # Extract columns
        beam_cols = [f'beam_{i}' for i in range(width)]
        grid_cols = [f'grid_{i}' for i in range(width)]
        next_cols = [f'next_{i}' for i in range(width)]

        self.beams = df[beam_cols].values.astype(np.float32)
        self.grids = df[grid_cols].values.astype(np.float32)
        self.targets = df[next_cols].values.astype(np.float32)

        print(f"Loaded {len(self)} samples, width={width}")

    def __len__(self):
        return len(self.beams)

    def __getitem__(self, idx):
        # Stack beams and grids as 2 channels: [2, width]
        x = np.stack([self.beams[idx], self.grids[idx]], axis=0)
        y = self.targets[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


class BeamCNN(nn.Module):
    """Small Conv1D network for beam propagation prediction.

    Input shape: [batch, 2, width] (2 channels: beams + grid)
    Output shape: [batch, width] (probability of beam at each position)
    """

    def __init__(self, width: int = 16):
        super().__init__()
        self.width = width

        # Conv layers with padding to maintain width
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, 2, width]
        x = self.relu(self.conv1(x))  # [batch, 8, width]
        x = self.relu(self.conv2(x))  # [batch, 8, width]
        x = self.sigmoid(self.conv3(x))  # [batch, 1, width]
        return x.squeeze(1)  # [batch, width]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """Train the model with BCE loss."""
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                val_loss += criterion(output, y).item()
                # Accuracy: prediction > 0.5 matches target
                pred = (output > 0.5).float()
                val_acc += (pred == y).float().mean().item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # Restore best model
    model.load_state_dict(best_state)
    return model


def quantize_weights(model, bits=8):
    """Quantize weights to INT8 for FPGA implementation.

    Returns dict of quantized weights and scale factors.
    """
    quantized = {}

    for name, param in model.named_parameters():
        weights = param.data.cpu().numpy()

        # Symmetric quantization: scale to [-127, 127]
        max_abs = np.abs(weights).max()
        scale = (2**(bits-1) - 1) / max_abs if max_abs > 0 else 1.0

        q_weights = np.round(weights * scale).astype(np.int8)

        quantized[name] = {
            'weights': q_weights,
            'scale': scale,
            'shape': weights.shape
        }

        print(f"{name}: shape={weights.shape}, scale={scale:.4f}, "
              f"range=[{q_weights.min()}, {q_weights.max()}]")

    return quantized


def export_ocaml(quantized: dict, output_path: str, width: int):
    """Export quantized weights as OCaml module for Hardcaml."""

    lines = [
        "(* Auto-generated INT8 weights for beam CNN *)",
        "(* DO NOT EDIT - regenerate with train.py *)",
        "",
        "open Core",
        "",
        f"let width = {width}",
        "",
    ]

    for name, data in quantized.items():
        weights = data['weights']
        scale = data['scale']
        shape = data['shape']

        # Flatten and convert to OCaml array
        flat = weights.flatten().tolist()
        ocaml_name = name.replace('.', '_')

        lines.append(f"(* {name}: shape={shape}, scale={scale:.6f} *)")
        lines.append(f"let {ocaml_name}_scale = {scale:.6f}")
        lines.append(f"let {ocaml_name} = [|")

        # Format as rows of 8 values
        for i in range(0, len(flat), 8):
            row = flat[i:i+8]
            row_str = "; ".join(str(v) for v in row)
            if i + 8 < len(flat):
                lines.append(f"  {row_str};")
            else:
                lines.append(f"  {row_str}")

        lines.append("|]")
        lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Wrote OCaml weights to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train beam propagation CNN')
    parser.add_argument('--data', type=str, default='training/data.csv',
                        help='Path to training data CSV')
    parser.add_argument('--width', type=int, default=16,
                        help='Grid width')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--output', type=str, default='training/weights.ml',
                        help='Output path for OCaml weights')
    parser.add_argument('--model-path', type=str, default='training/model.pt',
                        help='Path to save PyTorch model')
    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    dataset = BeamDataset(args.data, width=args.width)

    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    model = BeamCNN(width=args.width)
    print(f"Model parameters: {model.count_parameters()}")

    # Train
    model = train_model(model, train_loader, val_loader,
                        epochs=args.epochs, lr=args.lr, device=device)

    # Save PyTorch model
    torch.save(model.state_dict(), args.model_path)
    print(f"Saved model to {args.model_path}")

    # Quantize and export
    quantized = quantize_weights(model)
    export_ocaml(quantized, args.output, args.width)

    # Test accuracy on full dataset
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=256):
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = (output > 0.5).float()
            correct += (pred == y).sum().item()
            total += y.numel()

    print(f"\nFinal accuracy: {100*correct/total:.2f}%")


if __name__ == '__main__':
    main()

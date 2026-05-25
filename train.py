"""
Training and evaluation script for DPCFFormer with correlation caching.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from model import DPCFFormer
from preprocessing import preprocess_patch, precompute_correlation_cache
from utils import load_dataset, mirror_hsi, metrics_from_confusion, save_classification_map, load_correlation_cache


def parse_args():
    parser = argparse.ArgumentParser("DPCFFormer for HSI Classification")
    parser.add_argument('--gpu', type=str, default='0', help='GPU index')
    parser.add_argument('--dataset', type=str, choices=['IN', 'PU', 'SV', 'HC'], default='IN')
    parser.add_argument('--train_ratio', type=float, default=0.01, help='Training ratio per class')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--patch_size', type=int, default=7, help='SCSP patch size')
    parser.add_argument('--hpf_patch', type=int, default=9, help='HPF sub-patch size')
    parser.add_argument('--top_k', type=int, default=13, help='Number of selected pixels')
    parser.add_argument('--group_size', type=int, default=4, help='GPCF group size')
    parser.add_argument('--num_encoders', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--mlp_dim', type=int, default=64, help='Hidden dimension in FFN')
    parser.add_argument('--group_pool_scale', type=int, default=4, help='Final group pooling scale')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save outputs')
    parser.add_argument('--cache_dir', type=str, default='./corr_cache', help='Directory to store correlation cache')
    return parser.parse_args()


def main():
    args = parse_args()

    args.dataset = 'IN'
    args.train_ratio = 0.01
    args.epochs = 20
    args.batch_size = 32

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    data, gt, num_classes, train_pos, test_pos, train_labels, test_labels = load_dataset(
        args.dataset, args.train_ratio)
    h, w, c = data.shape
    print(f"Data shape: {h}x{w}x{c}, classes: {num_classes}, train samples: {len(train_pos)}, test samples: {len(test_pos)}")

    # Create mirrored image
    mirror_img, pad = mirror_hsi(data, args.patch_size)
    print(f"Mirror image shape: {mirror_img.shape}, pad: {pad}")

    # Build cache path based on parameters
    cache_filename = f"{args.dataset}_patch{args.patch_size}_hpf{args.hpf_patch}_topk{args.top_k}.pkl"
    cache_path = os.path.join(args.cache_dir, cache_filename)

    # Precompute or load correlation cache for all non-background pixels
    if os.path.exists(cache_path):
        cache = load_correlation_cache(cache_path)
        # Verify that all training/test pixels are in cache
        missing = [tuple(p) for p in train_pos if tuple(p) not in cache] + \
                  [tuple(p) for p in test_pos if tuple(p) not in cache]
        if missing:
            print(f"Warning: {len(missing)} positions missing from cache, recomputing...")
            cache = precompute_correlation_cache(data, gt, args.patch_size, args.hpf_patch,
                                                 args.top_k, cache_path)
    else:
        print("Precomputing correlation cache...")
        cache = precompute_correlation_cache(data, gt, args.patch_size, args.hpf_patch,
                                             args.top_k, cache_path)

    # Preprocess all training and test patches using cached correlations
    print("Preprocessing patches with SCSP and GPCF (using cache)...")
    def preprocess_all(positions, use_cache=True):
        sequences = []
        for (x, y) in positions:
            seq = preprocess_patch(mirror_img, pad, (x, y),
                                   args.patch_size, args.hpf_patch,
                                   args.top_k, args.group_size,
                                   cache=cache if use_cache else None)
            # seq shape: (num_groups, 4*c)
            sequences.append(seq)
        # Stack all groups for all samples
        return np.concatenate(sequences, axis=0)

    # Training data: each sample expands to C(12, group_size) groups
    from itertools import combinations
    num_groups = len(list(combinations(range(12), args.group_size)))
    train_sequences = preprocess_all(train_pos, use_cache=True)
    train_labels_expanded = np.repeat(train_labels, num_groups)

    # Test data: we take only the first group for each sample (or can average across groups)
    test_sequences = np.array([preprocess_patch(mirror_img, pad, (x, y),
                                                args.patch_size, args.hpf_patch,
                                                args.top_k, args.group_size,
                                                cache=cache)[0]
                               for (x, y) in test_pos])

    print(f"Train sequences shape: {train_sequences.shape}, Test sequences shape: {test_sequences.shape}")

    # Convert to torch tensors
    train_x = torch.from_numpy(train_sequences).float()
    train_y = torch.from_numpy(train_labels_expanded).long()
    test_x = torch.from_numpy(test_sequences).float()
    test_y = torch.from_numpy(test_labels).long()

    train_dataset = Data.TensorDataset(train_x, train_y)
    test_dataset = Data.TensorDataset(test_x, test_y)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Build model
    model = DPCFFormer(
        num_classes=num_classes,
        spectral_bands=c,
        num_encoders=args.num_encoders,
        mlp_dim=args.mlp_dim,
        dropout=0.0,
        group_pool_scale=args.group_pool_scale
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 10, gamma=0.9)

    # Training loop
    print("Starting training...")
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            _, pred = output.max(1)
            correct += pred.eq(batch_y).sum().item()
            total += batch_y.size(0)
        train_acc = 100. * correct / total
        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                _, pred = output.max(1)
                correct += pred.eq(batch_y).sum().item()
                total += batch_y.size(0)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        val_acc = 100. * correct / total
        print(f"Epoch {epoch:3d} | Loss: {total_loss / total:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.dataset}_best.pth'))

    # Final evaluation on test set (compute detailed metrics)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f'{args.dataset}_best.pth')))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            _, pred = output.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    oa, aa, kappa, pa = metrics_from_confusion(conf_mat)
    print("\n========== Final Results ==========")
    print(f"Overall Accuracy (OA): {oa * 100:.2f}%")
    print(f"Average Accuracy (AA): {aa * 100:.2f}%")
    print(f"Kappa Coefficient: {kappa:.4f}")
    print("Per-class accuracy:")
    for i, acc in enumerate(pa):
        print(f"  Class {i + 1}: {acc * 100:.2f}%")

    # Generate classification map for entire image (using cache)
    print("Generating classification map...")
    pred_map = np.zeros((h, w), dtype=np.int32)
    all_nonzero = np.argwhere(gt > 0)
    print(f"Classifying {len(all_nonzero)} pixels...")
    for idx, (i, j) in enumerate(all_nonzero):
        seq = preprocess_patch(mirror_img, pad, (i, j),
                               args.patch_size, args.hpf_patch,
                               args.top_k, args.group_size,
                               cache=cache)[0]  # take first group
        seq_t = torch.from_numpy(seq).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(seq_t)
            pred = out.argmax(dim=1).item()
        pred_map[i, j] = pred + 1  # class labels start from 1
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx+1}/{len(all_nonzero)} pixels")
    save_path = os.path.join(args.save_dir, f'{args.dataset}_map.png')
    save_classification_map(pred_map, save_path)
    print(f"Map saved to {save_path}")


if __name__ == '__main__':
    main()
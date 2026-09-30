


import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def load_embeddings_and_labels(emb_path, lab_path, device):
    """
    Exactly apke code jaisa embeddings load karna
    """
    print(f"Loading embeddings from: {emb_path}")
    embeddings = np.load(emb_path)

    print(f"Loading labels from: {lab_path}")
    df_labels = pd.read_csv(lab_path)

    # Handle if labels have brackets (apke code jaisa)
    if df_labels['log_fluorescence'].dtype == object:
        df_labels['log_fluorescence'] = df_labels['log_fluorescence'].astype(str).str.replace('[','', regex=False).str.replace(']','', regex=False).astype(float)

    labels = df_labels['log_fluorescence'].values

    # Normalize labels to [0,1] (apke code jaisa)
    y_min, y_max = labels.min(), labels.max()
    normalized_labels = (labels - y_min) / (y_max - y_min)

    N_sequences, emb_dim = embeddings.shape
    print(f"✅ Loaded: {N_sequences} sequences, embedding dim={emb_dim}")
    print(f"Labels normalized from [{y_min:.4f}, {y_max:.4f}] to [0,1]")

    # Train/val/test split (apke code jaisa: 80/10/10)
    all_idxs = np.arange(N_sequences)
    train_idxs, temp_idxs = train_test_split(all_idxs, test_size=0.2, random_state=1234)
    val_idxs, test_idxs = train_test_split(temp_idxs, test_size=0.5, random_state=1234)

    print(f"Split: Train {len(train_idxs)} ({len(train_idxs)/N_sequences:.1%}), "
          f"Val {len(val_idxs)} ({len(val_idxs)/N_sequences:.1%}), "
          f"Test {len(test_idxs)} ({len(test_idxs)/N_sequences:.1%})")

    # Convert to tensors
    X_all = torch.tensor(embeddings, dtype=torch.float32, device=device)
    mean_emb = torch.mean(X_all[train_idxs], dim=0, keepdim=True)

    return {
        'X_all': X_all,
        'normalized_labels': normalized_labels,
        'train_idxs': train_idxs,
        'val_idxs': val_idxs,
        'test_idxs': test_idxs,
        'mean_emb': mean_emb,
        'emb_dim': emb_dim,
        'N_sequences': N_sequences,
        'y_min': y_min,
        'y_max': y_max
    }

def setup_domain(J, dt, Tmax):
    """
    Domain setup exactly apke code jaisa
    """
    a, b = 0.0, 1.0
    Nx = J + 1
    x_grid = np.linspace(a, b, Nx).reshape(-1, 1)
    t_grid = np.arange(0.0, Tmax + 1e-12, dt).reshape(-1, 1)
    Nt = t_grid.shape[0]

    # Initial condition - Gaussian (Eq. 5 from paper)
    A, x0, sigma = 0.3, 0.5, 0.1
    u0_x = A * np.exp(- (x_grid - x0)**2 / (2 * sigma**2))
    u0_x = np.clip(u0_x, 0.0, 1.0)

    print(f"Domain: Nx={Nx}, Nt={Nt}, dx={(b-a)/J:.5f}, dt={dt}")

    return {
        'x_grid': x_grid,
        't_grid': t_grid,
        'Nx': Nx,
        'Nt': Nt,
        'u0_x': u0_x,
        'a': a,
        'b': b
    }

print("✅ data_utils.py created successfully!")
"""Synthetic data mixing: sample generated rows and append to real training set."""

import os
import numpy as np
import pandas as pd


def load_generated_rows(gen_xls, ratio, original_train_len, seed=1234):
    """Sample generated rows from gen_xls at the given ratio.

    Args:
        gen_xls: path to the generated data Excel (sheets: batch_0..batch_9)
        ratio: float, number of generated samples = original_train_len * ratio
        original_train_len: size of the real training set
        seed: random seed for reproducibility

    Returns:
        sampled DataFrame or None, info dict
    """
    if ratio <= 0 or not gen_xls or not os.path.exists(gen_xls):
        return None, {'enabled': False}

    xls = pd.ExcelFile(gen_xls)
    frames = []
    for s in xls.sheet_names:
        if s.startswith('batch_'):
            df = pd.read_excel(xls, sheet_name=s)
            frames.append(df)
    if not frames:
        return None, {'enabled': False, 'reason': 'no_batch_sheets'}

    gen_df = pd.concat(frames, ignore_index=True)

    # required = ['path', 'y', 'm']
    # for col in required:
    #     if col not in gen_df.columns:
    #         return None, {'enabled': False, 'reason': f'missing_col_{col}'}

    # gen_df = gen_df.dropna(subset=['path', 'y', 'm']).copy()
    gen_df['path'] = gen_df['path'].astype(str)
    gen_df['y'] = gen_df['y'].astype(int)
    gen_df['m'] = gen_df['m'].astype(int)

    requested_n = int(round(original_train_len * ratio))
    available_n = len(gen_df)
    if requested_n <= 0 or available_n <= 0:
        return None, {'enabled': False, 'reason': 'empty'}

    rng = np.random.RandomState(seed)
    replace = requested_n > available_n
    sampled = gen_df.sample(n=requested_n, replace=replace, random_state=rng)
    sampled = sampled.reset_index(drop=True)

    info = {
        'enabled': True,
        'ratio': ratio,
        'requested': requested_n,
        'available': available_n,
        'sampled': len(sampled),
        'oversampled': replace,
        'label_dist': sampled['y'].value_counts().to_dict(),
    }
    print(f"[gen-mix] sampled {len(sampled)} generated rows "
          f"(ratio={ratio}, available={available_n}, oversample={replace})")
    return sampled, info


def extend_dataset(real_ds, gen_ds):
    """In-place append gen_ds lists onto real_ds."""
    n_real = len(real_ds.samples)
    n_gen = len(gen_ds.samples)
    real_ds.samples.extend(gen_ds.samples)
    real_ds.labels.extend(gen_ds.labels)
    real_ds.A_1.extend(gen_ds.A_1)
    real_ds.A_2.extend(gen_ds.A_2)
    real_ds.Machine.extend(gen_ds.Machine)
    real_ds.Hospital.extend(gen_ds.Hospital)
    print(f"[gen-mix] train dataset: real={n_real} + gen={n_gen} = {n_real + n_gen}")
    return n_gen

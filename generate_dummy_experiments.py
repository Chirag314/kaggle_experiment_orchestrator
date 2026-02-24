import random

import numpy as np
import pandas as pd

N = 100

model_types = [
    "LightGBM",
    "XGBoost",
    "CatBoost",
    "RandomForest",
    "LogisticRegression",
    "NeuralNet",
    "SVM",
    "KNN",
]

feature_variants = [
    "baseline",
    "with_target_encoding",
    "with_interactions",
    "with_freq_encoding",
    "with_text_features",
    "with_polynomials",
    "with_scaling",
    "with_domain_features",
    "with_image_embeddings",
    "with_time_features",
]

note_variants = [
    "ok",
    "improved",
    "bad_seed",
    "try_again",
    "stable",
    "high_leak_suspected",
    "strange_cv_split",
    "overfit_heavy",
    "too_slow",
    "great_lb_but_unstable",
]


def random_features():
    return random.choice(feature_variants)


def random_params(model):
    return f"{model}_hp_set_{random.randint(1, 15)}"


rows = []

for i in range(N):
    model = random.choice(model_types)

    # EXTREME diversity in CV
    # some terrible (0.5), some insane (0.95)
    cv = float(np.round(random.uniform(0.50, 0.95), 4))

    # EXTREME diversity in gap: big overfit or underfit
    # gap = cv - holdout, so positive = overfit, negative = underfit
    gap = float(np.round(random.uniform(-0.08, 0.10), 4))
    holdout = float(np.round(cv - gap, 4))

    # EXTREME diversity in train times: from <1s to 20min
    train_time = float(np.round(random.uniform(0.2, 1200.0), 2))

    rows.append(
        {
            "experiment_id": f"exp_{i:03d}",
            "model_type": model,
            "cv_metric": cv,
            "holdout_metric": holdout,
            "train_time_seconds": train_time,
            "features_desc": random_features(),
            "params_summary": random_params(model),
            "notes": random.choice(note_variants),
        }
    )

df = pd.DataFrame(rows)
df.to_csv("sample_experiments.csv", index=False)

df.head()

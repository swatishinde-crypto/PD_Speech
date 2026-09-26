"""Models.

Baseline
    SVM on the same features -- reproduces the published protocol so the
    proposed model is measured against a fair, honest reference.

Proposed: SAMTN (Severity-Aware Multi-Task Network)
    One shared encoder feeding three heads.

      diagnosis head   binary PD / HC                        (the task)
      severity head    ordinal UPDRS-II-5 regression         (auxiliary)
      speaker head     37-way identity, behind a gradient
                       reversal layer                        (adversarial)

    Why each head earns its place:

    * The severity head is defined for *every* subject -- controls are 0,
      ID31 is 1 -- so it supplies a continuous degradation target on all
      1000+ segments. It pushes the encoder to represent *how much* the
      voice is degraded rather than *which bucket* it fell in, which is
      what lets the model say something about the 7 patients whose speech
      clinicians rated as normal.

    * The speaker head is trained to fail. Gradients are reversed on the way
      into the encoder, so the encoder is actively penalised for retaining
      speaker identity. With 37 speakers and ~30 segments each, identity is
      the shortcut that produces the field's 99% numbers; removing it is
      what makes the remaining signal attributable to the disease.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config


# Grid searched over in the inner loop. This is the same family of settings
# the published protocol tunes over, so the baseline is a fair one rather
# than a strawman we can easily beat.
SVM_GRID = {
    "svm__C": [0.1, 1.0, 10.0, 100.0],
    "svm__gamma": ["scale", 1e-3, 1e-2, 1e-1],
    "svm__kernel": ["rbf", "linear"],
}


def make_svm_baseline(tune=True, groups=None, n_inner=5):
    """SVM baseline, optionally with a nested grid search.

    Two-stage, for speed. `probability=True` makes SVC run an internal 5-fold
    Platt calibration on every single fit, so putting it inside a grid search
    multiplies the work by ~5 per grid point -- with 32 points and 5 inner
    folds that is ~800 fits per outer fold instead of 1. Instead we:

        stage 1  grid search with probability=False, scored on
                 decision_function via ROC AUC (rank-based, so calibration
                 is irrelevant to model selection)
        stage 2  refit ONE calibrated SVC with the winning parameters

    Same selected model, roughly 160x less compute.

    The inner splits are grouped by subject as well as the outer ones. Tuning
    with a subject-blind inner loop would select hyperparameters using leaked
    information, making the outer leave-one-subject-out score optimistic even
    though the outer split itself is clean.
    """
    if not tune:
        return Pipeline([
            ("scale", StandardScaler()),
            ("svm", SVC(C=10.0, gamma="scale", kernel="rbf",
                        probability=True, class_weight="balanced",
                        random_state=config.SEED)),
        ])
    return _TunedSVM(groups=groups, n_inner=n_inner)


class _TunedSVM:
    """Minimal fit/predict_proba wrapper implementing the two-stage search."""

    def __init__(self, groups=None, n_inner=5):
        self.groups = groups
        self.n_inner = n_inner
        self.best_params_ = None
        self._model = None

    def fit(self, X, y, groups=None):
        groups = self.groups if groups is None else groups

        search_pipe = Pipeline([
            ("scale", StandardScaler()),
            ("svm", SVC(probability=False, class_weight="balanced",
                        random_state=config.SEED)),
        ])
        if groups is not None:
            cv = GroupKFold(n_splits=min(self.n_inner, len(set(groups))))
            split = cv.split(X, y, groups)
        else:
            cv = StratifiedKFold(n_splits=self.n_inner, shuffle=True,
                                 random_state=config.SEED)
            split = cv.split(X, y)

        gs = GridSearchCV(search_pipe, SVM_GRID, scoring="roc_auc",
                          cv=list(split), n_jobs=-1, refit=False)
        gs.fit(X, y)
        self.best_params_ = gs.best_params_

        # Stage 2: one calibrated fit with the winning parameters.
        final = Pipeline([
            ("scale", StandardScaler()),
            ("svm", SVC(probability=True, class_weight="balanced",
                        random_state=config.SEED)),
        ])
        final.set_params(**gs.best_params_)
        self._model = final.fit(X, y)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        return self._model.predict(X)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class SAMTN(nn.Module):
    def __init__(self, n_features, n_speakers, hidden=None, dropout=None):
        super().__init__()
        hidden = hidden or config.HIDDEN
        dropout = config.DROPOUT if dropout is None else dropout

        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        z = hidden // 2
        self.diagnosis = nn.Linear(z, 1)
        self.severity = nn.Sequential(nn.Linear(z, 32), nn.ReLU(), nn.Linear(32, 1))
        self.speaker = nn.Sequential(nn.Linear(z, 64), nn.ReLU(), nn.Linear(64, n_speakers))

    def forward(self, x, grl_alpha=0.0):
        z = self.encoder(x)
        spk = self.speaker(GradientReversal.apply(z, grl_alpha))
        return self.diagnosis(z).squeeze(-1), self.severity(z).squeeze(-1), spk, z


def train_samtn(Xtr, ytr, sevtr, spktr, n_speakers, *,
                use_severity=True, use_adversary=True, device="cpu", verbose=False):
    """Fit one SAMTN. Ablation flags switch the auxiliary heads off."""
    torch.manual_seed(config.SEED)
    model = SAMTN(Xtr.shape[1], n_speakers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LR,
                            weight_decay=config.WEIGHT_DECAY)

    pos = float(ytr.sum())
    pos_weight = torch.tensor([(len(ytr) - pos) / max(pos, 1.0)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()

    X = torch.tensor(Xtr, dtype=torch.float32, device=device)
    y = torch.tensor(ytr, dtype=torch.float32, device=device)
    s = torch.tensor(sevtr, dtype=torch.float32, device=device)
    k = torch.tensor(spktr, dtype=torch.long, device=device)

    n = len(X)
    for epoch in range(config.EPOCHS):
        model.train()
        # Ramp the reversal in: a strong adversary from step 0 stops the
        # encoder learning anything at all.
        p = epoch / max(config.EPOCHS - 1, 1)
        alpha = config.GRL_ALPHA * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0) if use_adversary else 0.0

        perm = torch.randperm(n, device=device)
        for i in range(0, n, config.BATCH_SIZE):
            idx = perm[i:i + config.BATCH_SIZE]
            if len(idx) < 2:          # BatchNorm needs >1 sample
                continue
            d, sv, sp, _ = model(X[idx], alpha)
            loss = bce(d, y[idx])
            if use_severity:
                loss = loss + config.LAMBDA_SEVERITY * mse(sv, s[idx])
            if use_adversary:
                loss = loss + config.LAMBDA_ADVERSARY * ce(sp, k[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  loss {loss.item():.4f}  alpha {alpha:.2f}")
    return model


@torch.no_grad()
def predict(model, X, device="cpu"):
    model.eval()
    d, sv, _, _ = model(torch.tensor(X, dtype=torch.float32, device=device), 0.0)
    return torch.sigmoid(d).cpu().numpy(), sv.cpu().numpy()

"""Hidden Markov Model for regime transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hmmlearn import hmm


@dataclass
class HMMConfig:
    """HMM configuration."""

    n_regimes: int = 4
    covariance_type: str = "full"
    n_iter: int = 100
    random_state: int = 42


class RegimeHMM:
    """Gaussian HMM for regime detection and transition modeling.

    Fits a Gaussian HMM to features (typically returns + volatility)
    and provides regime labels with transition probabilities.
    """

    def __init__(self, config: HMMConfig | None = None):
        self.config = config or HMMConfig()
        self.model = hmm.GaussianHMM(
            n_components=self.config.n_regimes,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            random_state=self.config.random_state,
        )
        self._fitted = False

    def fit(self, features: np.ndarray) -> np.ndarray:
        """Fit HMM and return most likely state sequence.

        Args:
            features: (n_samples, n_features) observation array.
                      Typical features: returns, volatility, volume.

        Returns:
            (n_samples,) integer regime labels via Viterbi decoding.
        """
        self.model.fit(features)
        self._fitted = True
        return self.model.predict(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict regime labels for new observations.

        Args:
            features: (n_samples, n_features) observation array.

        Returns:
            (n_samples,) integer regime labels.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get regime membership probabilities.

        Returns:
            (n_samples, n_regimes) probability matrix.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(features)

    @property
    def transition_matrix(self) -> np.ndarray:
        """Get fitted transition probability matrix.

        Returns:
            (n_regimes, n_regimes) transition matrix where [i, j] is
            P(regime_j at t+1 | regime_i at t).
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.transmat_

    @property
    def stationary_distribution(self) -> np.ndarray:
        """Get stationary distribution of regimes.

        Computes the true stationary distribution as the left eigenvector
        of the transition matrix with eigenvalue 1.

        Returns:
            (n_regimes,) stationary probabilities.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        transmat = self.model.transmat_
        # Stationary distribution: pi @ T = pi, sum(pi) = 1
        # Equivalent to left eigenvector with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
        # Find eigenvector closest to eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = np.abs(stationary)  # Ensure non-negative (sign ambiguity from solver)
        stationary = stationary / stationary.sum()  # Normalize
        return stationary

    @property
    def regime_means(self) -> np.ndarray:
        """Get mean observation vector for each regime.

        Returns:
            (n_regimes, n_features) means.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.means_

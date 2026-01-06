#### Port of sklearn.linear_model.RidgeCV with slimmed down data validation steps
# Data validation create a huge memory overhead when working with large model features
# In order to make RidgeCV more memory efficient, we assume Brain-Score only passes valid 
# data into this class
# Additionally, we add caching of the eigendecomposition using Brain-Score's result caching system.
# Source of original: https://github.com/scikit-learn/scikit-learn/blob/71cfff335/sklearn/linear_model/_ridge.py

import torch
import numpy as np
from scipy import linalg, sparse
import xarray as xr
import functools

from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import _check_sample_weight, validate_data
from sklearn.linear_model._base import _preprocess_data, _rescale_data
from sklearn.linear_model._ridge import _RidgeGCV, _check_gcv_mode

import logging
logger = logging.getLogger(__name__)

from result_caching import store



class RidgeGCVCPU(_RidgeGCV):

    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        *,
        fit_intercept=True,
        scoring=None,
        copy_X=False,
        gcv_mode=None,
        store_cv_results=False,
        is_clf=False,
        alpha_per_target=False,
        skip_check_array=True,
        validate_during_preprocessing=False,
        eigh_driver='evr', # e.g. 'evd', 'evr', ...
    ):
        super().__init__(
            alphas=alphas,
            fit_intercept=fit_intercept,
            scoring=scoring,
            copy_X=copy_X,
            gcv_mode=gcv_mode,
            store_cv_results=store_cv_results,
            is_clf=is_clf,
            alpha_per_target=alpha_per_target,
        )
        # NEW PARAMS regulating data validation behavior
        self.skip_check_array = skip_check_array # skip expensive validation of the input array 
        self.validate_during_preprocessing = validate_during_preprocessing # whether to validate the data again after it has been preprocessed
        self.eigh_driver = eigh_driver # driver to use for scipy.linalg.eigh (evr is best, evd is faster)

    def __repr__(self):
            """
            Custom repr to keep cache key filename short.
            """
            return f"{self.__class__.__name__}(object at {id(self)})"

    def validate_cache_identifiers(self, fitting_kwargs):
        """
        Validate that all required identifying parameters are present and not None for caching
        """ 
        
        # Check if result is cached based on the benchmark's identifying parameters passed in fitting_kwargs
        required_cache_params = {'model_id', 'stimuli_identifier', 'number_of_trials', 'require_variance', 'benchmark_id'}

        provided_params = set(fitting_kwargs.keys())
        missing_params = required_cache_params - provided_params
        
        if missing_params:
            raise ValueError(
                f"Caching requires all identifying parameters {required_cache_params}. "
                f"Missing: {missing_params}. Provided: {provided_params}"
            )
        for param in required_cache_params:
            if fitting_kwargs[param] is None:
                raise ValueError(
                    f"Caching requires identifying parameter '{param}' to be not None. "
                    f"Got None."
                )

    def decompose_eigs(self, X, sqrt_sw):
        """
        This part of the decomposition does not depend on y and can be reused for different neuroids.
        Adapted from _eigen_decompose_gram and moved to a separate function to enable caching its results.
        """

        # if X is dense it has already been centered in preprocessing
        K, X_mean = self._compute_gram(X, sqrt_sw)
        if self.fit_intercept:
            # to emulate centering X with sample weights,
            # ie removing the weighted average, we add a column
            # containing the square roots of the sample weights.
            # by centering, it is orthogonal to the other columns
            K += np.outer(sqrt_sw, sqrt_sw)
        logger.info("getting eigvals")

        if torch.cuda.is_available():
            logger.info("  using GPU for eigh computation")
            K_tensor = torch.from_numpy(K).to(device='cuda')
            eigvals_tensor, Q_tensor = torch.linalg.eigh(K_tensor)
            eigvals = eigvals_tensor.cpu().numpy()
            Q = Q_tensor.cpu().numpy()
            torch.cuda.empty_cache()
        else:
            logger.info("  using CPU for eigh computation")
            eigvals, Q = linalg.eigh(K, driver=self.eigh_driver)
        return X_mean, eigvals, Q

    
    @store(identifier_ignore=['X', 'sqrt_sw'])
    def cache_decomposition(self, X, sqrt_sw,
                            model_id,
                            stimuli_identifier,
                            number_of_trials,
                            require_variance,
                            benchmark_id,
                            ):
        return self.decompose_eigs(X, sqrt_sw)

    def _eigen_decompose_gram(self, X, y, sqrt_sw, fitting_kwargs=None):
        """
        Eigendecomposition of X.X^T, used when n_samples <= n_features.
        """
        
        if fitting_kwargs is None:
            logger.info("No fitting_kwargs provided - caching disabled, computing decomposition directly")
            X_mean, eigvals, Q = self.decompose_eigs(X, sqrt_sw)
        else:
            self.validate_cache_identifiers(fitting_kwargs)

            logger.info(f"Caching enabled - decomposing with fitting_kwargs: {fitting_kwargs}")
            X_mean, eigvals, Q = self.cache_decomposition(X, sqrt_sw, **fitting_kwargs)
            logger.info(f"Decompose completed")
            
        QT_y = np.dot(Q.T, y)
        logger.info("got eigvals")
        return X_mean, eigvals, Q, QT_y
    
    
    def svd_decompose(self, X, sqrt_sw):
        X_mean = np.zeros(X.shape[1], dtype=X.dtype)
        if self.fit_intercept:
            # to emulate fit_intercept=True situation, add a column
            # containing the square roots of the sample weights
            # by centering, the other columns are orthogonal to that one
            intercept_column = sqrt_sw[:, None]
            X = np.hstack((X, intercept_column))
        logger.info("Computing SVD of X")
        U, singvals, _ = linalg.svd(X, full_matrices=0)
        singvals_sq = singvals**2
        
        return X, X_mean, singvals_sq, U
    
    @store(identifier_ignore=['X', 'sqrt_sw'])
    def cache_svd(self, X, sqrt_sw,
                            model_id,
                            stimuli_identifier,
                            number_of_trials,
                            require_variance,
                            benchmark_id,
                            ):
        return self.svd_decompose(X, sqrt_sw)
    
    
    def _svd_decompose_design_matrix(self, X, y, sqrt_sw, fitting_kwargs=None):
        # X already centered
        if fitting_kwargs is None:
            logger.info("No fitting_kwargs provided - caching disabled, computing SVD directly")
            X, X_mean, singvals_sq, U = self.svd_decompose(X, sqrt_sw)
        else:
            self.validate_cache_identifiers(fitting_kwargs)
            X, X_mean, singvals_sq, U = self.cache_svd(X, sqrt_sw, **fitting_kwargs)
        UT_y = np.dot(U.T, y)
        logger.info("SVD computed")
        return X_mean, singvals_sq, U, UT_y
    



    def fit(self, X, y, sample_weight=None, score_params=None, fitting_kwargs=None, internal_test_source=None, internal_test_target=None):
        """Fit Ridge regression model with gcv.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training data. Will be cast to float64 if necessary.

        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to float64 if necessary.

        sample_weight : float or ndarray of shape (n_samples,), default=None
            Individual weights for each sample. If given a float, every sample
            will have the same weight. Note that the scale of `sample_weight`
            has an impact on the loss; i.e. multiplying all weights by `k`
            is equivalent to setting `alpha / k`.

        score_params : dict, default=None
            Parameters to be passed to the underlying scorer.

            .. versionadded:: 1.5
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
        """
        logger.info(f"Fitting with skip_check_array={self.skip_check_array} and " +\
                    f"validate_during_preprocessing={self.validate_during_preprocessing} and " +\
                    f"copy_X={self.copy_X}")
        
        #### Convert Brain-Score's DataArrays 
        if isinstance(X, xr.DataArray):
            logger.info("Converting DataArray X")
            X = X.data 
            logger.info("X converted")
        if isinstance(y, xr.DataArray):
            logger.info("Converting DataArray y")
            y = y.data
            logger.info("y converted")
        
        # Ensure arrays are writable (DataArray.data may be read-only)
        if not X.flags.writeable:
            logger.info("X is read-only, making a copy")
            X = X.copy()
        if not y.flags.writeable:
            logger.info("y is read-only, making a copy")
            y = y.copy()
        
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=["csr", "csc", "coo"],
            dtype=[np.float64],
            multi_output=True,
            y_numeric=True,
            skip_check_array=self.skip_check_array,
        )
        logger.info("Data validated")
        # alpha_per_target cannot be used in classifier mode. All subclasses
        # of _RidgeGCV that are classifiers keep alpha_per_target at its
        # default value: False, so the condition below should never happen.
        assert not (self.is_clf and self.alpha_per_target)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        self.alphas = np.asarray(self.alphas)
        
        unscaled_y = y
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept, #TODO: confirm this is correct
            copy=self.copy_X,
            sample_weight=sample_weight,
            check_input=self.validate_during_preprocessing,
        )
        logger.info(f"Successfully preprocessed X with shape {X.shape} and y with shape {y.shape}")
        gcv_mode = _check_gcv_mode(X, self.gcv_mode)

        if gcv_mode == "eigen":
            logger.info("Using eigen decomposition for GCV")
            decompose = functools.partial(self._eigen_decompose_gram, fitting_kwargs=fitting_kwargs)
            solve = self._solve_eigen_gram
        elif gcv_mode == "svd":
            logger.info("Using SVD decomposition for GCV")
            if sparse.issparse(X):
                logger.info("  and X is sparse")
                decompose = self._eigen_decompose_covariance
                solve = self._solve_eigen_covariance
            else:
                logger.info("  and X is dense")
                decompose = self._svd_decompose_design_matrix
                solve = self._solve_svd_design_matrix

        n_samples = X.shape[0]

        if sample_weight is not None:
            X, y, sqrt_sw = _rescale_data(X, y, sample_weight)
        else:
            sqrt_sw = np.ones(n_samples, dtype=X.dtype)
            logger.info(f"No sample weights found, so created {sqrt_sw.dtype} with shape {sqrt_sw.shape}")


        X_mean, *decomposition = decompose(X, y, sqrt_sw)

        n_y = 1 if len(y.shape) == 1 else y.shape[1]
        n_alphas = 1 if np.ndim(self.alphas) == 0 else len(self.alphas)

        if self.store_cv_results:
            self.cv_results_ = np.empty((n_samples * n_y, n_alphas), dtype=X.dtype)

        best_coef, best_score, best_alpha = None, None, None
        
        logger.info(f"Starting loop over alphas")
        for i, alpha in enumerate(np.atleast_1d(self.alphas)):
            logger.info(f"  alpha = {alpha}")
            G_inverse_diag, c = solve(float(alpha), y, sqrt_sw, X_mean, *decomposition)
            if self.scoring is None:
                squared_errors = (c / G_inverse_diag) ** 2
                alpha_score = self._score_without_scorer(squared_errors=squared_errors)
                if self.store_cv_results:
                    self.cv_results_[:, i] = squared_errors.ravel()
            else:
                predictions = y - (c / G_inverse_diag)
                # Rescale predictions back to original scale
                if sample_weight is not None:  # avoid the unnecessary division by ones
                    if predictions.ndim > 1:
                        predictions /= sqrt_sw[:, None]
                    else:
                        predictions /= sqrt_sw
                predictions += y_offset

                if self.store_cv_results:
                    self.cv_results_[:, i] = predictions.ravel()

                score_params = score_params or {}
                alpha_score = self._score(
                    predictions=predictions,
                    y=unscaled_y,
                    n_y=n_y,
                    scorer=self.scoring,
                    score_params=score_params,
                )

            # Keep track of the best model
            if best_score is None:
                # initialize
                if self.alpha_per_target and n_y > 1:
                    best_coef = c
                    best_score = np.atleast_1d(alpha_score)
                    best_alpha = np.full(n_y, alpha)
                else:
                    best_coef = c
                    best_score = alpha_score
                    best_alpha = alpha
            else:
                # update
                if self.alpha_per_target and n_y > 1:
                    to_update = alpha_score > best_score
                    best_coef[:, to_update] = c[:, to_update]
                    best_score[to_update] = alpha_score[to_update]
                    best_alpha[to_update] = alpha
                elif alpha_score > best_score:
                    best_coef, best_score, best_alpha = c, alpha_score, alpha

        logger.info("Alpha loop finished")
        self.alpha_ = best_alpha
        self.best_score_ = best_score
        self.dual_coef_ = best_coef
        self.coef_ = safe_sparse_dot(self.dual_coef_.T, X)
        if y.ndim == 1 or y.shape[1] == 1:
            self.coef_ = self.coef_.ravel()

        if sparse.issparse(X):
            X_offset = X_mean * X_scale
        else:
            X_offset += X_mean * X_scale
        self._set_intercept(X_offset, y_offset, X_scale)

        if self.store_cv_results:
            if len(y.shape) == 1:
                cv_results_shape = n_samples, n_alphas
            else:
                cv_results_shape = n_samples, n_y, n_alphas
            self.cv_results_ = self.cv_results_.reshape(cv_results_shape)

        return self
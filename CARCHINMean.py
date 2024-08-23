from __future__ import annotations
from copy import deepcopy
from typing import Any, Callable, cast
import warnings
import numpy as np
from arch.typing import (
    ArrayLike1D,
    DateLike,
    Float64Array,
    Literal,
)
from arch.univariate.volatility import ConstantVariance
from arch.utility.array import ensure1d
from arch.utility.exceptions import (
    ConvergenceWarning,
    StartingValueWarning,
    convergence_warning,
    starting_value_warning,
)
from arch.univariate import ARCHInMean
from arch.univariate.base import ARCHModelResult


_callback_info = {"iter": 0, "llf": 0.0, "count": 0, "display": 1}


def _callback(parameters: Float64Array, *args: Any) -> None:
    """
    Callback for use in optimization

    Parameters
    ----------
    parameters : ndarray
        Parameter value (not used by function).
    *args
        Any other arguments passed to the minimizer.

    Notes
    -----
    Uses global values to track iteration, iteration display frequency,
    log likelihood and function count
    """

    _callback_info["iter"] += 1
    disp = "Iteration: {0:>6},   Func. Count: {1:>6.3g},   Neg. LLF: {2}"
    if _callback_info["iter"] % _callback_info["display"] == 0:
        print(
            disp.format(
                _callback_info["iter"], _callback_info["count"], _callback_info["llf"]
            )
        )


def constraint(a: Float64Array, b: Float64Array) -> list[dict[str, object]]:
    """
    Generate constraints from arrays

    Parameters
    ----------
    a : ndarray
        Parameter loadings
    b : ndarray
        Constraint bounds

    Returns
    -------
    constraints : dict
        Dictionary of inequality constraints, one for each row of a

    Notes
    -----
    Parameter constraints satisfy a.dot(parameters) - b >= 0
    """

    def factory(coeff: Float64Array, val: float) -> Callable[..., float]:
        def f(params: Float64Array, *args: Any) -> float:
            return np.dot(coeff, params) - val

        return f

    constraints = []
    for i in range(a.shape[0]):
        con = {"type": "ineq", "fun": factory(a[i], b[i])}
        constraints.append(con)

    return constraints


class CustomARCHInMean(ARCHInMean):
    def fit(
            self,
            arch_mean_type: Literal["standard", "half_var"] = "half_var",
            update_freq: int = 1,
            disp: bool | Literal["off", "final"] = "final",
            starting_values: ArrayLike1D | None = None,
            cov_type: Literal["robust", "classic"] = "robust",
            show_warning: bool = True,
            first_obs: int | DateLike | None = None,
            last_obs: int | DateLike | None = None,
            tol: float | None = None,
            options: dict[str, Any] | None = None,
            backcast: None | float | Float64Array = None,
    ) -> ARCHModelResult:
        if self._y_original is None:
            raise RuntimeError("Cannot estimate model without data.")
        # 1. Check in ARCH or Non-normal dist.  If no ARCH and normal,
        # use closed form
        v, d = self.volatility, self.distribution
        offsets = np.array((self.num_params, v.num_params, d.num_params), dtype=int)
        total_params = sum(offsets)

        # Closed form is applicable when model has no parameters
        # Or when distribution is normal and constant variance
        has_closed_form = (
                v.closed_form and d.num_params == 0 and isinstance(v, ConstantVariance)
        )

        self._adjust_sample(first_obs, last_obs)

        resids = np.asarray(self.resids(self.starting_values()), dtype=float)
        self._check_scale(resids)
        if self.scale != 1.0:
            # Scale changed, rescale data and reset model
            self._y = cast(np.ndarray, self.scale * np.asarray(self._y_original))
            self._scale_changed()
            self._adjust_sample(first_obs, last_obs)
            resids = np.asarray(self.resids(self.starting_values()), dtype=float)

        if backcast is None:
            backcast = v.backcast(resids)
        else:
            assert backcast is not None
            backcast = v.backcast_transform(backcast)

        if has_closed_form:
            try:
                return self._fit_no_arch_normal_errors(cov_type=cov_type)
            except NotImplementedError:
                pass
        assert backcast is not None
        if total_params == 0:
            return self._fit_parameterless_model(cov_type=cov_type, backcast=backcast)

        sigma2 = np.zeros_like(resids)
        self._backcast = backcast
        sv_volatility = v.starting_values(resids)
        self._var_bounds = var_bounds = v.variance_bounds(resids)
        v.compute_variance(sv_volatility, resids, sigma2, backcast, var_bounds)
        std_resids = resids / np.sqrt(sigma2)

        # 2. Construct constraint matrices from all models and distribution
        constraints = (
            self.constraints(),
            self.volatility.constraints(),
            self.distribution.constraints(),
        )
        num_cons = []
        for c in constraints:
            assert c is not None
            num_cons.append(c[0].shape[0])
        num_constraints = np.array(num_cons, dtype=int)
        num_params = offsets.sum()
        a = np.zeros((int(num_constraints.sum()), int(num_params)))
        b = np.zeros(int(num_constraints.sum()))

        for i, c in enumerate(constraints):
            assert c is not None
            r_en = num_constraints[: i + 1].sum()
            c_en = offsets[: i + 1].sum()
            r_st = r_en - num_constraints[i]
            c_st = c_en - offsets[i]

            if r_en - r_st > 0:
                a[r_st:r_en, c_st:c_en] = c[0]
                b[r_st:r_en] = c[1]
        bounds = self.bounds()
        if arch_mean_type == 'standard':
            bounds = self.bounds()
        elif arch_mean_type == 'half_var':
            bounds = [(-np.inf, np.inf), (-0.5, -0.5)]
        bounds.extend(v.bounds(resids))
        bounds.extend(d.bounds(std_resids))

        # 3. Construct starting values from all models
        sv = starting_values
        if starting_values is not None:
            assert sv is not None
            sv = ensure1d(sv, "starting_values")
            valid = sv.shape[0] == num_params
            if a.shape[0] > 0:
                satisfies_constraints = a.dot(sv) - b >= 0
                valid = valid and satisfies_constraints.all()
            for i, bound in enumerate(bounds):
                valid = valid and bound[0] <= sv[i] <= bound[1]
            if not valid:
                warnings.warn(starting_value_warning, StartingValueWarning)
                starting_values = None

        if starting_values is None:
            sv = np.hstack(
                [self.starting_values(), sv_volatility, d.starting_values(std_resids)]
            )

        # 4. Estimate models using constrained optimization
        _callback_info["count"], _callback_info["iter"] = 0, 0
        if not isinstance(disp, str):
            disp = bool(disp)
            disp = "off" if not disp else "final"
        if update_freq <= 0 or disp == "off":
            _callback_info["display"] = 2 ** 31

        else:
            _callback_info["display"] = update_freq
        disp_flag = True if disp == "final" else False

        func = self._loglikelihood
        args = (sigma2, backcast, var_bounds)
        ineq_constraints = constraint(a, b)

        from scipy.optimize import minimize

        options = {} if options is None else options
        options.setdefault("disp", disp_flag)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "Values in x were outside bounds during a minimize step",
                RuntimeWarning,
            )
            opt = minimize(
                func,
                sv,
                args=args,
                method="SLSQP",
                bounds=bounds,
                constraints=ineq_constraints,
                tol=tol,
                callback=_callback,
                options=options,
            )

        if show_warning:
            warnings.filterwarnings("always", "", ConvergenceWarning)
        else:
            warnings.filterwarnings("ignore", "", ConvergenceWarning)

        if opt.status != 0 and show_warning:
            warnings.warn(
                convergence_warning.format(code=opt.status, string_message=opt.message),
                ConvergenceWarning,
            )

        # 5. Return results
        params = opt.x
        loglikelihood = -1.0 * opt.fun

        mp, vp, dp = self._parse_parameters(params)

        resids = np.asarray(self.resids(mp), dtype=float)
        vol = np.zeros_like(resids)
        self.volatility.compute_variance(vp, resids, vol, backcast, var_bounds)
        vol = cast(Float64Array, np.sqrt(vol))

        try:
            r2 = self._r2(mp)
        except NotImplementedError:
            r2 = np.nan

        names = self._all_parameter_names()
        # Reshape resids and vol
        first_obs, last_obs = self._fit_indices
        resids_final = np.empty_like(self._y, dtype=np.float64)
        resids_final.fill(np.nan)
        resids_final[first_obs:last_obs] = resids
        vol_final = np.empty_like(self._y, dtype=np.float64)
        vol_final.fill(np.nan)
        vol_final[first_obs:last_obs] = vol

        fit_start, fit_stop = self._fit_indices
        model_copy = deepcopy(self)
        assert isinstance(r2, float)

        return ARCHModelResult(
            params,
            None,
            r2,
            resids_final,
            vol_final,
            cov_type,
            self._y_series,
            names,
            loglikelihood,
            self._is_pandas,
            opt,
            fit_start,
            fit_stop,
            model_copy,
        )
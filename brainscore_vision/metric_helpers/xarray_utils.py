import numpy as np
import xarray as xr
from collections import defaultdict, deque

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly, array_is_element, walk_coords
from brainscore_vision.metric_helpers import Defaults
from brainscore_vision.metrics import Score

import logging
logger = logging.getLogger(__name__)

class XarrayRegression:
    """
    Adds alignment-checking, un- and re-packaging, and comparison functionality to a regression.
    """

    def __init__(self, regression, expected_dims=Defaults.expected_dims, neuroid_dim=Defaults.neuroid_dim,
                 neuroid_coord=Defaults.neuroid_coord, stimulus_dim=Defaults.stimulus_dim, stimulus_coord=Defaults.stimulus_coord):
        self._regression = regression
        self._expected_dims = expected_dims
        self._neuroid_dim = neuroid_dim
        self._neuroid_coord = neuroid_coord
        self._stimulus_dim = stimulus_dim
        self._stimulus_coord = stimulus_coord
        self._target_neuroid_values = None

    def fit(self, source, target, **kwargs):
        logger.debug("Aligning source and target assemblies...")
        source, target = self._align(source), self._align(target)
        target = self._align_target_with_source(source, target)
        logger.debug("Assemblies aligned, XarrayRegression calls fit")

        self._regression.fit(source, target, **kwargs)

        self._target_neuroid_values = {}
        for name, dims, values in walk_coords(target):
            if self._neuroid_dim in dims:
                assert array_is_element(dims, self._neuroid_dim)
                self._target_neuroid_values[name] = values

    def _align_target_with_source(self, source, target):
        """
        Reorder the target assembly to match the order of presentations in the source assembly.
        This replaces the old way of securing alignment, which sorted both arrays:
        source, target = source.sortby(self._stimulus_coord), target.sortby(self._stimulus_coord)
        If source is a large array from a DNN with many features, sorting can be slow and memory-intensive.
        This method handles duplicates, if we are sure there are no duplicates we could optimize further.
        """
        
        
        s_idx = source.coords[self._stimulus_coord].to_index()
        t_idx = target.coords[self._stimulus_coord].to_index()

        if len(s_idx) != len(t_idx):
            raise ValueError(
                f"Assemblies must have same number of presentations"
            )

        # Map label -> queue of positions where it appears in target_index
        positions_in_target = defaultdict(deque)
        for pos, label in enumerate(t_idx):
            positions_in_target[label].append(pos)

        # For each label in source, pop one unused matching position from target
        reordering = []
        for label in s_idx:
            q = positions_in_target.get(label)
            if not q:
                raise KeyError(
                    f"Cannot match label {label!r}: target_index is missing it or duplicates do not match"
                )
            reordering.append(q.popleft())

        # Ensure we used every target position exactly once
        leftovers = {k: len(v) for k, v in positions_in_target.items() if len(v) > 0}
        if leftovers:
            raise ValueError(
                "Found unmatched labels while matching assemblies for regression"
            )

        return target.isel({self._stimulus_dim: reordering})

    def predict(self, source):
        source = self._align(source)
        predicted_values = self._regression.predict(source)
        prediction = self._package_prediction(predicted_values, source=source)
        return prediction

    def _package_prediction(self, predicted_values, source):
        coords = {coord: (dims, values) for coord, dims, values in walk_coords(source)
                  if not array_is_element(dims, self._neuroid_dim)}
        # re-package neuroid coords
        dims = source.dims
        # if there is only one neuroid coordinate, it would get discarded and the dimension would be used as coordinate.
        # to avoid this, we can build the assembly first and then stack on the neuroid dimension.
        neuroid_level_dim = None
        if len(self._target_neuroid_values) == 1:  # extract single key: https://stackoverflow.com/a/20145927/2225200
            (neuroid_level_dim, _), = self._target_neuroid_values.items()
            dims = [dim if dim != self._neuroid_dim else neuroid_level_dim for dim in dims]
        for target_coord, target_value in self._target_neuroid_values.items():
            # this might overwrite values which is okay
            coords[target_coord] = (neuroid_level_dim or self._neuroid_dim), target_value
        prediction = NeuroidAssembly(predicted_values, coords=coords, dims=dims)
        if neuroid_level_dim:
            prediction = prediction.stack(**{self._neuroid_dim: [neuroid_level_dim]})

        return prediction

    def _align(self, assembly):
        assert set(assembly.dims) == set(self._expected_dims), \
            f"Expected {set(self._expected_dims)}, but got {set(assembly.dims)}"
        return assembly.transpose(*self._expected_dims)


class XarrayCorrelation:
    def __init__(self, correlation, correlation_coord=Defaults.stimulus_coord, neuroid_coord=Defaults.neuroid_coord):
        self._correlation = correlation
        self._correlation_coord = correlation_coord
        self._neuroid_coord = neuroid_coord

    def __call__(self, prediction, target):
        # align
        prediction = prediction.sortby([self._correlation_coord, self._neuroid_coord])
        target = target.sortby([self._correlation_coord, self._neuroid_coord])
        assert np.array(prediction[self._correlation_coord].values == target[self._correlation_coord].values).all()
        assert np.array(prediction[self._neuroid_coord].values == target[self._neuroid_coord].values).all()
        # compute correlation per neuroid
        neuroid_dims = target[self._neuroid_coord].dims
        assert len(neuroid_dims) == 1
        correlations = []
        for i, coord_value in enumerate(target[self._neuroid_coord].values):
            target_neuroids = target.isel(**{neuroid_dims[0]: i})  # `isel` is about 10x faster than `sel`
            prediction_neuroids = prediction.isel(**{neuroid_dims[0]: i})
            r, p = self._correlation(target_neuroids, prediction_neuroids)
            correlations.append(r)
        # package
        result = Score(correlations,
                       coords={coord: (dims, values)
                               for coord, dims, values in walk_coords(target) if dims == neuroid_dims},
                       dims=neuroid_dims)
        return result


# ops that also applies to attrs (and attrs of attrs), which are xarrays
def recursive_op(*arrs, op=lambda x:x):
    # the attrs structure of each arr must be the same
    val = op(*arrs)
    attrs = arrs[0].attrs
    for attr in attrs:
        attr_val = arrs[0].attrs[attr]
        if isinstance(attr_val, xr.DataArray):
            attr_arrs = [arr.attrs[attr] for arr in arrs]
            attr_val = recursive_op(*attr_arrs, op=op)
        val.attrs[attr] = attr_val
    return val


# apply a callable to every slice of the xarray along the specified dimensions
def apply_over_dims(callable, *asms, dims, njobs=-1):
    asms = [asm.transpose(*dims, ...) for asm in asms]
    sizes = [asms[0].sizes[dim] for dim in dims]

    def apply_helper(sizes, dims, *asms):
        xarr = []
        attrs = {}
        size = sizes[0]
        rsizes = sizes[1:]
        dim = dims[0]
        rdims = dims[1:]

        if len(sizes) == 1:
            # parallel execution on the last applied dimension
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=njobs)(delayed(callable)(*[asm.isel({dim:s}) for asm in asms]) for s in range(size))
        else:
            results = []
            for s in range(size):
                arr = apply_helper(rsizes, rdims, *[asm.isel({dim:s}) for asm in asms])
                results.append(arr)

        for arr in results:
            if arr is not None:
                for k,v in arr.attrs.items():
                    assert isinstance(v, xr.DataArray)
                    attrs.setdefault(k, []).append(v.expand_dims(dim))
                xarr.append(arr)
       
        if not xarr:
            return
        else:
            xarr = xr.concat(xarr, dim=dim)
            attrs = {k: xr.concat(vs, dim=dim) for k,vs in attrs.items()}
            xarr.coords[dim] = asms[0].coords[dim]
            for k,v in attrs.items():
                attrs[k].coords[dim] = asms[0].coords[dim]
                xarr.attrs[k] = attrs[k]
            return xarr

    return apply_helper(sizes, dims, *asms)
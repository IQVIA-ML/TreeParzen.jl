# Configuration

The optimizer settings are provided through a `Config` object, or as keyword arguments to `fmin`.

- `threshold::Float64`: probability threshold used to model the expected improvement criterion; between `0` and `1`.
- `linear_forgetting::Int`: number of recent trials used for probabilistic modeling; older trials are linearly down-weighted.
- `draws::Int`: number of candidates sampled when selecting the next optimization point.
- `random_trials::Int`: number of random candidate trials before TreeParzen optimization begins.
- `prior_weight::Float64`: relative weight of the user-specified prior parameters and the trial history; between `0` and `1`.

See [Getting Started](usage.md) for examples of `Config` with the ask/tell API.
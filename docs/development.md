# Development

## Diagnostics

The `inside()` diagnostic prints an `AbstractDelayed` or `History` object as a tree in the console.

Each child of `AbstractDelayed` contains a random identifier. This allows distinct Julia objects with equal contents to remain distinct keys, matching Python dictionary behavior. The identifiers are hidden by `inside()`.

## Tests

Run the test suite from the repository root:

```bash
julia --project -e "using Pkg; Pkg.test()"
```

For contribution and pull request guidance, see the [contributing guidelines](https://github.com/IQVIA-ML/TreeParzen.jl/blob/master/CONTRIBUTING.md).
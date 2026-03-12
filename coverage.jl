# Run with `julia --project=<path/to/Project.toml> coverage.jl`
# e.g. `julia --project=. coverage.jl`
using Pkg
using Coverage

Pkg.activate(".")

# Defaults to src/; alternatively, supply the folder name as argument
coverage = process_folder()
# Get total coverage for all Julia files
covered_lines, total_lines = get_summary(coverage)
# Or process a single file
# @show get_summary(process_file("src/MyPkg.jl"))
println("Covered lines | ", covered_lines)
println("Total lines   | ", total_lines)
println("Coverage      | ", round(covered_lines / total_lines * 100; digits = 1), "%")

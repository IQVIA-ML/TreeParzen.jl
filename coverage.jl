# Run with `julia --project=<path/to/Project.toml> coverage.jl`
# e.g. `julia --project=. coverage.jl`
using Pkg
using Coverage

Pkg.activate(".")

# If Coverage is not installed correctly, it can fail but still exit without a
# proper error code, which causes the calling script to report success. So we
# wrap it in a try.
if !isdir(Pkg.dir("Coverage"))
    throw(LoadError(string("Coverage", 4, "Coverage is not installed correctly")))
end

# defaults to src/; alternatively, supply the folder name as argument
coverage = process_folder()
# Get total coverage for all Julia files
covered_lines, total_lines = get_summary(coverage)
# Or process a single file
# @show get_summary(process_file("src/MyPkg.jl"))
println("Covered lines | ", covered_lines)
println("Total lines   | ", total_lines)
println("Coverage      | ", round(covered_lines / total_lines * 100; digits = 1), "%")

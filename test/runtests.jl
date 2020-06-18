using Test

@time @testset "Unit Tests" begin
    @testset "Small functions" begin

        @info "Bincount"
        @test include("bincount.jl")

        @info "Configuration"
        @test include("configuration.jl")

        @info "dfs"
        @test include("dfs.jl")

        @info "graph"
        @test include("graph.jl")

        @info "forgettingweights"
        @test include("forgettingweights.jl")

        # Needs translation from Python
        # include("adaptive_parzen_normal_orig.jl")

        @info "Trials"
        @test include("trials.jl")

        @info "GMM1"
        @test include("gmm.jl")

        @info "GMM1 Math and QGMM1 Math"
        @test include("gmm_math.jl")

        @info "LGMM1"
        @test include("lgmm.jl")

        @info "Resolve"
        include("resolvenodes.jl")

        @info "Spaces"
        include("spaces.jl")
    end

    @testset "Larger tests" begin
        @info "Basic"
        @test include("basic.jl")

        @info "bjkomer/Squared"
        @test include("bjkomer/squared.jl")

        @info "bjkomer/Function fitting"
        @test include("bjkomer/function_fitting.jl")

        @info "Official Cases"
        @test include("official_cases.jl")

        @info "fmin/Quadratic"
        @test include("fmin/quadratic.jl")

        @info "fmin/Return Inf"
        @test include("fmin/return_inf.jl")

        @info "fmin/Submit points to Trial"
        @test include("fmin/points.jl")

        @info "Silvrback"
        @test include("silvrback.jl")

        @info "Vooban/Basic"
        @test include("vooban/basic.jl")

        @info "Vooban/Find min"
        @test include("vooban/find_min.jl")

        @info "Vooban/Status Fail skip"
        @test include("vooban/status_fail_skip.jl")
    end

    @testset "Samplers" begin
        @info "hp_pchoice"
        @test include("hp.jl")

        @info "LogQuantNormal"
        @test include("logquantnormal.jl")

        @info "LogUniform"
        @test include("loguniform.jl")

        @info "QuantUniform"
        @test include("quantuniform.jl")

        @info "QuantNormal"
        @test include("quantnormal.jl")

        @info "LogQuantUniform"
        @test include("logquantuniform.jl")

        @info "Uniform"
        include("uniform.jl")
    end

    @testset "MLJ" begin
        @info "MLJ Unit tests"
        @test include("MLJ/unit.jl")

        @info "MLJ integration"
        @test include("MLJ/integration.jl")
    end

    @info "API"
    @test include("api.jl")

    # Run this test last so that the print output is just above the test report
    @info "SpacePrint"
    @test include("spaceprint.jl")
end

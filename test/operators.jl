module Testoperators

using Test
import TreeParzen


CONFIG = TreeParzen.Config()


@testset "Delayed binary operators -- addition" begin

    choicevals = [1, 2, 3]

    a = TreeParzen.HP.Choice(:a, choicevals)
    b = TreeParzen.HP.Choice(:b, choicevals)

    expr = a + b

    trials = TreeParzen.Trials.Trial[]

    for i in 1:100
        trial = TreeParzen.ask(expr)
        TreeParzen.tell!(trials, trial, 1.)
        @test 2 <= trial.hyperparams <= 6
    end


    for i in 1:1000
        trial = TreeParzen.ask(expr, trials, CONFIG)
        @test 2 <= trial.hyperparams <= 6
    end


end


@testset "Delayed binary operators -- subtraction" begin

    choicevals = [1, 2, 3]

    a = TreeParzen.HP.Choice(:a, choicevals)
    b = TreeParzen.HP.Choice(:b, choicevals)

    expr = a - b

    trials = TreeParzen.Trials.Trial[]

    for i in 1:100
        trial = TreeParzen.ask(expr)
        TreeParzen.tell!(trials, trial, 1.)
        @test -2 <= trial.hyperparams <= 2
    end


    for i in 1:1000
        trial = TreeParzen.ask(expr, trials, CONFIG)
        @test -2 <= trial.hyperparams <= 2
    end


end


@testset "Delayed binary operators -- multiplication" begin

    choicevals = [1, 2, 3]

    a = TreeParzen.HP.Choice(:a, choicevals)
    b = TreeParzen.HP.Choice(:b, choicevals)

    expr = a * b

    trials = TreeParzen.Trials.Trial[]

    for i in 1:100
        trial = TreeParzen.ask(expr)
        TreeParzen.tell!(trials, trial, 1.)
        @test 1 <= trial.hyperparams <= 9
    end


    for i in 1:1000
        trial = TreeParzen.ask(expr, trials, CONFIG)
        @test 1 <= trial.hyperparams <= 9
    end


end


@testset "Delayed binary operators -- division" begin

    choicevals = [1, 2, 3]

    a = TreeParzen.HP.Choice(:a, choicevals)
    b = TreeParzen.HP.Choice(:b, choicevals)

    expr = a / b

    trials = TreeParzen.Trials.Trial[]

    for i in 1:100
        trial = TreeParzen.ask(expr)
        TreeParzen.tell!(trials, trial, 1.)
        @test 0.333 <= trial.hyperparams <= 3
    end


    for i in 1:1000
        trial = TreeParzen.ask(expr, trials, CONFIG)
        @test 0.333 <= trial.hyperparams <= 3
    end


end


@testset "Delayed binary operators -- modulus" begin

    choicevals = [1, 2, 3]

    a = TreeParzen.HP.Choice(:a, choicevals)
    b = TreeParzen.HP.Choice(:b, choicevals)

    expr = a % b

    trials = TreeParzen.Trials.Trial[]

    for i in 1:100
        trial = TreeParzen.ask(expr)
        TreeParzen.tell!(trials, trial, 1.)
        @test 0 <= trial.hyperparams <= 2
    end


    for i in 1:1000
        trial = TreeParzen.ask(expr, trials, CONFIG)
        @test 0 <= trial.hyperparams <= 2
    end


end


@testset "Delayed binary operators -- power" begin

    choicevals = [1, 2, 3]

    a = TreeParzen.HP.Choice(:a, choicevals)
    b = TreeParzen.HP.Choice(:b, choicevals)

    expr = a ^ b

    trials = TreeParzen.Trials.Trial[]

    for i in 1:100
        trial = TreeParzen.ask(expr)
        TreeParzen.tell!(trials, trial, 1.)
        @test 1 <= trial.hyperparams <= 27
    end


    for i in 1:1000
        trial = TreeParzen.ask(expr, trials, CONFIG)
        @test 1 <= trial.hyperparams <= 27
    end


end


@testset "Delayed unary operators -- subtraction" begin

    choicevals = [1, 2, 3]

    a = TreeParzen.HP.Choice(:a, choicevals)

    expr = -a

    trials = TreeParzen.Trials.Trial[]

    for i in 1:100
        trial = TreeParzen.ask(expr)
        TreeParzen.tell!(trials, trial, 1.)
        @test -3 <= trial.hyperparams <= -1
    end


    for i in 1:1000
        trial = TreeParzen.ask(expr, trials, CONFIG)
        @test -3 <= trial.hyperparams <= -1
    end


end


@testset "Delayed unary operators -- user-specified (ceil)" begin

    a = TreeParzen.HP.Uniform(:a, 0., 1.)
    expr = TreeParzen.Delayed.UnaryOperator(a, ceil)


    trials = TreeParzen.Trials.Trial[]

    for i in 1:100
        trial = TreeParzen.ask(expr)
        TreeParzen.tell!(trials, trial, 1.)
        @test trial.hyperparams == 1
    end


    for i in 1:1000
        trial = TreeParzen.ask(expr, trials, CONFIG)
        @test trial.hyperparams == 1
    end


end

end # module
true

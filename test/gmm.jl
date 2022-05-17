module TestGMM

using Statistics
using Test
import TreeParzen: GMM

N_SAMPLES = 100_000

# see here for an expression of the variance of a gaussian mixture
# https://stats.stackexchange.com/a/16609
mixture_variance(weights, sigmas, means) = sum(weights .* (sigmas .^ 2)) + sum(weights .* (means .^ 2)) - sum(weights .* means) ^ 2

@testset "GMM1" begin

    # Check that mu is specified correctly
    mu_test_std = 1e-7
    mu_test_mu = 1e1
    mu_test_weight = 1e0
    # for whatever standard deviation we use we don't expect more than 1-2 orders of magnitude deviation from this
    # if sigma was larger the rtol might need adjusting upwards a little but not a lot.
    @test all(isapprox.(mu_test_mu, GMM.GMM1([mu_test_weight], [mu_test_mu], [mu_test_std], N_SAMPLES); rtol = mu_test_std * 1e1))

    # Check that sigma is specified correctly
    sigma_test_std = 1e1
    sigma_test_mu = 0e0
    sigma_test_weight = 1e0
    @test isapprox(std(GMM.GMM1([sigma_test_weight], [sigma_test_mu], [sigma_test_std], N_SAMPLES)), sigma_test_std; rtol=1e-1)

    # Check a mixture of gaussians, equally weighted, and contributes to mixture variance
    mixture_test_stds = [1e-6, 1e-6]
    mixture_test_mus = [0e0, 1e0]
    mixture_test_weights = [0.5e0, 0.5e0]
    samples = GMM.GMM1(mixture_test_weights, mixture_test_mus, mixture_test_stds, N_SAMPLES)
    expected_variance = mixture_variance(mixture_test_weights, mixture_test_stds, mixture_test_mus)
    expected_mean = sum(mixture_test_weights .* mixture_test_mus)
    @test isapprox(expected_mean, mean(samples); rtol=3e-2)
    @test isapprox(expected_variance, var(samples); rtol=1e-1)

    # Check an uneven weighted mixture has expected variance (this is quite a bit more variable)
    uneven_mixture_test_stds = [1e-6, 1e-6]
    uneven_mixture_test_mus = [0e0, 1e0]
    uneven_mixture_test_weights = [1 - 1e-4, 1e-4]
    samples = GMM.GMM1(uneven_mixture_test_weights, uneven_mixture_test_mus, uneven_mixture_test_stds, N_SAMPLES)
    expected_variance = mixture_variance(uneven_mixture_test_weights, uneven_mixture_test_stds, uneven_mixture_test_mus)
    expected_mean = sum(uneven_mixture_test_weights .* uneven_mixture_test_mus)
    @test size(samples) == (N_SAMPLES,)
    # these are whopper tolerances because although the numbers come out close
    # the highly unbalanced nature of the mixture makes drawn samples from run to run extremely variable
    # even with 1e5 samples drawn we don't expect most of the time to draw from the 2nd
    # distribution, but that can differ by maybe an ofer of magnitude from run to run
    @test isapprox(expected_mean, mean(samples); rtol=1e0)
    @test isapprox(expected_variance, var(samples); rtol=1e0)

    # lpdf scalar one component
    llval = GMM.GMM1_lpdf([1.0], [1.], [1.0], [2.0])
    @test size(llval) == (1,) # Shape should match first parameter above
    @test isapprox(llval, [log(1.0 / sqrt(2pi * 2.0 ^ 2))])

    # lpdf vector N components
    llval = GMM.GMM1_lpdf([1.0, 0.0], [0.25, 0.25, .5], [0.0, 1.0, 2.0], [1.0, 2.0, 5.0])

    a = .25 / sqrt(2pi * 1^2) * exp(-.5 * 1^2)
    a += .25 / sqrt(2pi * 2^2)
    a += .5 / sqrt(2pi * 5^2) * exp(-.5 * (1 / 5)^2)
    @test size(llval) == (2,)
    @test isapprox(llval[1], log(a))

    a = .25 / sqrt(2pi * 1^2)
    a += .25 / sqrt(2pi * 2^2) * exp(-.5 * (1 / 2)^2)
    a += .5 / sqrt(2pi * 5^2) * exp(-.5 * (2 / 5)^2)
    @test isapprox(llval[2], log(a))

    # weights, mus and sigmas have different lengths
    @test_throws DimensionMismatch GMM.GMM1([1.0, 0.0], [0.0, 1.0, 2.0], [10.0], 1000)

    # non-1 sum of weights throws
    @test_throws DomainError GMM.GMM1([1., 2.], [1., 2.], [1., 2.], 1)
    @test_throws DomainError GMM.GMM1([0.5, 0.6], [1., 2.], [1., 2.], 1)
    @test_throws DomainError GMM.GMM1([0.2, 0.1], [1., 2.], [1., 2.], 1)
    @test_throws DomainError GMM.GMM1([-1., 2.], [1., 2.], [1., 2.], 1)
    @test_throws DomainError GMM.GMM1([-0.5, -0.5], [1., 2.], [1., 2.], 1)

end

end
true

@inline const_term(n, μ) = (-1.0)^(n-1.0) * (n-1.0) * μ^n
@inline binom_term(n, k, μ, exp_gi) = binomial(n, k) * (-μ)^(n - k) * exp_gi

@inline function binom_sum(μ, exp_vals)
    m = length(exp_vals) + 1
    sum([binom_term(m, k+1, μ, v) for (k,v) in enumerate(exp_vals)]) + const_term(m)
end

# solve central moments problem of generic callable functions via MonteCarlo
function DiffEqBase.solve(cmprob::CentralMomentProblem, expalg::MonteCarlo) 
    params = parameters(cmprob.exp_prob)
    dist = distribution(cmprob.exp_prob)
    g = observable(cmprob.exp_prob)
    exp_set = mean(g(rand(dist), params) for _ ∈ 1:expalg.trajectories)

    results = []

    for (i,n) ∈ enumerate(cmprob.ns)

        obs_ind = 1

        if i != 1
            obs_ind = sum(cmprob.ns[1:i-1]) + 1
        end

        obs_stats = [zero(exp_set[1]), binom_sum(exp_set[obs_ind], exp_set[obs_ind:obs_ind+n-1])]
        push!(results, obs_stats)
    end

    return results

end

# solve expectation over DEProblem via MonteCarlo
function DiffEqBase.solve(cmprob::CentralMomentProblem{F}, expalg::MonteCarlo) where F<:SystemMap
    d = distribution(exprob)
    cov = input_cov(exprob)
    S = mapping(exprob)
    g = observable(exprob)

    prob_func = function (prob, i, repeat)
        u0, p = cov(rand(d), prob.u0, prob.p)
        remake(prob, u0=u0, p=p)
    end

    output_func(sol, i) = (g(sol,sol.prob.p), false)

    monte_prob = EnsembleProblem(S.prob;
                output_func=output_func,
                prob_func=prob_func)
    sol = solve(monte_prob, S.args...;trajectories=expalg.trajectories,S.kwargs...)
    exp_set = mean(g(sol.u))

    results = []

    for (i,n) ∈ enumerate(ns)

        obs_ind = 1

        if i != 1
            obs_ind = sum(ns[1:i-1]) + 1
        end

        obs_stats = [zero(exp_set[1]), binom_sum(exp_set[obs_ind], exp_set[obs_ind:obs_ind+n-1])]
        push!(results, obs_stats)
    end

    return results
end

# Solve Koopman expectation
function DiffEqBase.solve(cmprob::CentralMomentProblem, expalg::Koopman, args...; 
                        maxiters=1000000,
                        batch=0,
                        quadalg=HCubatureJL(),
                        ireltol=1e-2, iabstol=1e-2,
                        kwargs...) where {A<:AbstractExpectationADAlgorithm}

    prob = cmprob.exp_prob

    integrand = build_integrand(prob, expalg, Val(batch > 1))
    lb, ub = extrema(prob.d)

    sol = integrate(quadalg, expalg.sensealg, integrand, lb, ub, prob.params;
            reltol=ireltol, abstol=iabstol, maxiters=maxiters, 
            nout = prob.nout, batch = batch, 
            kwargs...)

    return sol 
end
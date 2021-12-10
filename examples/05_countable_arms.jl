include("../src/MAB.jl")
using .MAB
using Statistics
using Plots
using Distributions
using StatsBase

struct  CABEnv
    mean_rewards::Vector{Float64}
    sample_weights::StatsBase.FrequencyWeights
    total_time::Int
    function CABEnv(mean_rewards::Vector{Float64}, sample_weights::Vector{Float64}, total_time::Int) 
        new(mean_rewards,  FrequencyWeights(sample_weights), total_time)
    end
end

function sample_arms(env::CABEnv, size::Int)
    return sample(env.mean_rewards, env.sample_weights, size)
end

function alg_adaptively_greedy(env::CABEnv,init_size::Int, init_epsilon::Float64, alg)
    arms = []
    cumu_rewards = [0]
    # play each arm once

    init_means = sample_arms(env, init_size)
    for m in init_means
        push!(arms, Arms.Bernoulli(m))
    end

    for t = 1:env.total_time
        epislon = minimum([1, init_epsilon / t ]) 
        if t > init_size && rand(Bernoulli(epislon))
            # explore
            # println("prev: $(length(arms))")
            push!(arms, Arms.Bernoulli(sample_arms(env, 1)[1]))
            # println("next: $(length(arms))")
            add_arms!(alg, 1)
            # println(length(arms))
        end
        if get_arm_index(alg) > length(arms)
            # exploit
            println(get_arm_index(alg), length(arms))
        end
        reward = Arms.pull!(arms[get_arm_index(alg)])
        
        update_reward!(alg, reward)
        push!(cumu_rewards, last(cumu_rewards) + reward)
    end
    return arms, cumu_rewards[2:end]
end

function alg_fixed_set(env::CABEnv, candidate_size::Int, alg)
    arms = []
    cumu_rewards = [0]
    # play each arm once

    init_means = sample_arms(env, candidate_size)
    for m in init_means
        push!(arms, Arms.Bernoulli(m))
    end

    for t = 1:env.total_time
        reward = Arms.pull!(arms[get_arm_index(alg)])
        update_reward!(alg, reward)
        push!(cumu_rewards, last(cumu_rewards) + reward)
    end
    return arms, cumu_rewards[2:end]
end

mutable struct ArmInfo
    arm
    counts::Int
    mean_reward::Float64
end
function alg_elimination(env::CABEnv, init_size::Int, pull_times, decay_ratio)
    candidate_set = []
    init_means = sample_arms(env, init_size)
    for m in init_means
        push!(candidate_set, ArmInfo(Arms.Bernoulli(m), 0, 0))
    end

    round = 1
    t = 0
    cumu_rewards = [0]
    while round <= trunc(Int, env.total_time / pull_times / init_size) 
        for arm in candidate_set
            cumu_reward = 0
            for i = 1:pull_times
                reward  = Arms.pull!(arm.arm)
                cumu_reward += reward
                push!(cumu_rewards, last(cumu_rewards) + reward)
                t += 1
            end
            arm.mean_reward = (cumu_reward + arm.mean_reward * arm.counts )/ (pull_times + arm.counts)
            arm.counts += pull_times
        end
        if length(candidate_set) > 1
            sort!(candidate_set, by = x -> x.mean_reward, rev=true)
            candidate_set = candidate_set[1:trunc(Int, length(candidate_set) * (1 - 1 / (1 + round)))]
            next_means = sample_arms(env, trunc(Int, length(candidate_set) / (round + 1)))
            for m in next_means
                push!(candidate_set, ArmInfo(Arms.Bernoulli(m), 0, 0))
            end
        else
            break
        end

        round += 1
    end
    
    while t < env.total_time
        push!(cumu_rewards, last(cumu_rewards) + Arms.pull!(candidate_set[1].arm))
        t += 1
    end 

    return candidate_set, cumu_rewards[2:end]
end

total_time = 10000
mean_rewards = collect(0.1:0.2:0.9)
alpha = 0.2
env = CABEnv(mean_rewards, [(1-alpha)/(length(mean_rewards)-1) .* ones(length(mean_rewards)-1); alpha], total_time)
env = CABEnv(mean_rewards, [(1-alpha)/(length(mean_rewards)-1) .* ones(length(mean_rewards)-1); alpha], total_time)
candidate_size = trunc(Int, log(env.total_time) / alpha)
arms, rewards = alg_elimination(env,candidate_size, trunc(Int, log(total_time)), 1)
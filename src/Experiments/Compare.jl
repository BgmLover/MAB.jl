# Experiment compare mulitple algorithms
using Random
struct Tuple2{A,B}
    a::A
    b::B
    function Tuple2{Int,Int}(a,b)
        new(a,b)
    end
end

struct Compare{T1,T2 } <: BanditExpBase
    bandit::Vector{T1}
    algorithms::Vector{T2}

    # function Compare{T1<:Arms.BanditArmBase,T2<:BanditAlgorithmBase}(
    #     _bandit::Vector{T1},
    #     _algo::Vector{T2}
    #     )
    #     new{typeof(_bandit), typeof(_algo)}( _bandit, _algo )
    # end
end

function run( experiment::Compare, noOfTimeSteps::Integer, noOfRounds::Integer )

    result = Dict{String,Array{Float64,2}}()
    for alg ∈ experiment.algorithms
        Random.seed!( 1729 );  # "Magic" Seed initialization for RNG - across all algorithms
        observations = zeros( noOfTimeSteps, noOfRounds )
        for _round = 1:noOfRounds
            reset!( alg )
            # Reset arms of the bandit
            for arm ∈ experiment.bandit
                Arms.reset!( arm )
            end
            for _n = 1:noOfTimeSteps
                armToPull   = get_arm_index( alg )
                reward      = Arms.pull!( experiment.bandit[armToPull] )
                update_reward!( alg, reward )
                observations[_n,_round] = reward
                # Process tick() for all arms except the pulled arm
                for arm in experiment.bandit
                    if arm == experiment.bandit[armToPull]
                        continue
                    else
                        Arms.tick!( arm )
                    end
                end
            end
        end
        avgReward = mean( observations, dims = 2 )
        result[info_str(alg,true)] = avgReward
    end
    return result
end

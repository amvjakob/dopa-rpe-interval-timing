# Update the pacemaker rate η based on the bidirectional learning rule.
# Julia version of https://github.com/jgmikhael/flowoftime
# Written 21 Oct 2021 by AMVJ

"""
    TDeta(n, T, η, power, Vh, r, γ[; α=.1, numiter=1000])

Run TD learning model on pacemaker rate.
Returns the values of η and the estimated subjective reward times.

# Arguments
- `n::Integer`: number of states
- `T::Integer`: (objective) time of reward delivery
- `η::Float64`: initial pacemaker rate
- `power::Float64`: compression factor (y = η*t^power)
- `Vh::Vector{Float64}`: estimated value function
- `r::Float64`: reward magnitude
- `γ::Float64`: discount factor

- `α_η::Float64`: learning rate of η
- `numiter::Integer`: number of iterations
"""
function TDeta(n, T, η, power, Vh, r, γ; α_η=.008, numiter=100)    
    # housekeeping for reward schedule
    r0 = r
    r = zeros(n)

    # number of trials
    abs_t = numiter * n

    # subjective reward time
    Y = floor(η * T^power)  

    # dV/dy
    dVdy = [(diff(Vh[1:end-1]) + diff(Vh[2:end])) / 2; 0; 0] 

    # update eta
    ηs = zeros(abs_t)
    Ys = zeros(numiter)
    abs = 0

    # for each trial
    for iter in 1:numiter
        # for each timepoint within that trial    
        for yi in 1:n-1
            # only if haven't gotten reward yet
            if yi < Y+1          
                abs += 1

                # compute RPE
                δ = r[yi] + γ * Vh[yi+1] - Vh[yi]
                
                # update eta
                η = max(0, η + α_η * yi * dVdy[yi] * δ / η)

                # update Y and reward schedule
                Y = floor(Int, η * T^power)
                r = zeros(n); r[Y] = r0

                # store new eta
                ηs[abs] = η
            end
        end

        # trial is over, lick happened at yi = Y
        Ys[iter] = η * T^power
    end
    
    return ηs[ηs .!= 0], Ys
end

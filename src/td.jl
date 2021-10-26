# TD learning model.
# Julia version of https://github.com/jgmikhael/flowoftime
# Written 21 Oct 2021 by AMVJ

using Distributions

"""
    TD(n, Y, σ, r, γ[; α=.1, numiter=1000])

Run TD learning model.

# Arguments
- `n::Integer`: number of states
- `Y::Integer`: (subjective) time of reward delivery
- `σ::Float64`: width of features against subjective time
- `r::Float64`: reward magnitude
- `γ::Float64`: discount factor

- `α::Float64`: learning rate of w
- `numiter::Integer`: number of iterations
"""
function TD(n, Y, σ, r, γ; α=.1, numiter=1000)
    # subjective time
    y = collect(1:n)
    
    # housekeeping for reward schedule
    r0 = r
    r = zeros(n)
    r[Y] = r0

    # create x:  x[i,j] = (subjective time i, feature j)
    x = hcat([pdf.(dist, y) for dist in Normal.(y, σ)]...)

    # number of trials
    abs_t = numiter * n

    # initialize w, Vh, delta
    w = zeros(abs_t, n)
    Vh = zeros(n, numiter)
    δ = zeros(abs_t)

    # for each trial
    abs = 1
    for iter in 2:numiter
        # for each timepoint within that trial    
        for yi in 1:n-1
            abs += 1

            # compute Vh
            Vh[yi,iter] = w[abs-1,:]' * x[yi,:]  
            
            # compute RPE
            δ[abs] = r[yi] + γ * Vh[yi+1, iter-1] - Vh[yi,iter]

            # update w
            w[abs,:] = max.(w[abs-1,:] + α * δ[abs] * x[yi,:], 0)
        end
    end

    # record outputs
    w = w[round(Int, .9 * end), :]
    Vh = Vh[:, end-1]

    return Vh, w
end

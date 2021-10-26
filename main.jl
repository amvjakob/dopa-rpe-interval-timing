# Diverse tools to preprocess photometry data
# Written 21 Oct 2021 by AMVJ

import Base: string

using DSP       # signal smoothing
using MATLAB    # load raw photometry
using CSV       # store processed data
using GLM       # for slope computation

using Logging, DataFrames, Dates, Statistics

# Binning mode
"Parent type for different time-binning modes."
abstract type BinMode end
"Same bin size in each trial, number of bins may vary across trials."
struct AbsBin <: BinMode end
"Same number of bins in each trial, bin size may vary across trials."
struct RelBin <: BinMode end

string(b::AbsBin) = "absbin"
string(b::RelBin) = "relbin"

# Bin alignment
"Parent type for alignment of time-bins."
abstract type BinAlign end
"Align time-bins at trial start."
struct AlignStart <: BinAlign end
"Align time-bins at trial end."
struct AlignEnd   <: BinAlign end
"Align time-bins at lick time."
struct AlignLick  <: BinAlign end

string(b::AlignStart) = "align_start"
string(b::AlignEnd)   = "align_end"
string(b::AlignLick)  = "align_lick"


# Get time-bins for each possible configuration of bin modes and alignments
get_bintimes(cue, lick, nbins, binmode::AbsBin, binalign::AlignStart) =
    cue .+ collect(0:nbins) .* (TRIALLEN / nbins)

    get_bintimes(cue, lick, nbins, binmode::AbsBin, binalign::AlignEnd) =
    (cue + TRIALLEN) .- collect(nbins:-1:0) .* (TRIALLEN / nbins)

function get_bintimes(cue, lick, nbins, binmode::AbsBin, binalign::AlignLick)
    lim = div(nbins, 2)
    return lick .+ collect(-lim:lim) .* (TRIALLEN / nbins)
end

get_bintimes(cue, lick, nbins, binmode::RelBin, binalign::AlignStart) =
    cue .+ collect(0:nbins) .* (TRIALLEN / nbins)

get_bintimes(cue, lick, nbins, binmode::RelBin, binalign::AlignEnd) =
    (cue + TRIALLEN) .- collect(nbins:-1:0) .* (TRIALLEN / nbins)

function get_bintimes(cue, lick, nbins, binmode::RelBin, binalign::AlignLick)
    lim = div(nbins, 2)
    binlen_pre = (2.0 / nbins) * (lick - cue)
    binlen_post = (2.0 / nbins) * (cue + TRIALLEN - lick)
    return vcat(
        lick .+ collect(-lim:0) .* binlen_pre,
        lick .+ collect(1:lim) .* binlen_post
    )
end

"""
    fname = get_mat_fname(dir)

Get name of first `.mat` file in directory `dir`.
"""
function get_matfile_fname(dir)
    files = readdir(dir)
    matfiles = filter(f -> endswith(f, ".mat"), files)

    @assert length(matfiles) > 0

    return join(split(matfiles[1], ".")[1:end-1], ".")
end

"""
    fname, matfile = load_matfile(dir)

Load first `.mat` file in directory `dir`.
"""
function load_matfile(dir)
    fname = get_matfile_fname(dir)
    return fname, read_matfile("$dir/$fname.mat")
end

"""
    remove_outliers!(y[, n_std=15])

Remove points further away than `n_std` std deviations from `y` and replace them 
by interpolation between point before and after.
"""
function remove_outliers!(y::Vector{Float64}, n_std=15)
    μ, σ = mean(y), std(y)

    outliers = findall(abs.(y .- μ) .> n_std .* σ)
    for outlier in outliers
        a = μ
        a_index = outlier - 1
        for i in a_index:-1:1
            if i ∉ outliers
                a = y[i]
                a_index = i
                break
            end
        end

        b = μ
        b_index = outlier + 1
        for i in b_index:length(y)
            if i ∉ outliers
                b = y[i]
                b_index = i
                break
            end
        end

        y[outlier] = a + (b-a)/(b_index - a_index) * (outlier - a_index)
    end
    
    return y
end

"""
    moving_average(y, win)

Compute moving average of signal `y` with window length `win` (in samples).
"""
function moving_average(y::Vector{Float64}, win::Integer)
    avg = similar(y)
    
    # cumulative moving average
    @views cumsum!(avg[1:win], y[1:win])
    avg[1:win] ./= 1:win
    
    # moving average
    @views cumsum!(avg[win+1:end], 1.0/win .* (y[win+1:end] .- y[1:end-win]))
    avg[win+1:end] .+= avg[win]

    return avg
end


"""
    get_dF_F!(y, fs[, win=200])

Compute dF/F signal from raw photometry signal `y` with sampling rate `fs` by 
using a window length `win` (in seconds).
"""
function get_dF_F!(y::Vector{Float64}, fs, win=200)
    F0 = moving_average(y, round(Int, win * fs))
    
    y .-= F0
    y ./= F0
    
    return y
end

"""
    load_exclusions(dir)

Load list of excluded trials from session in `dir`.
"""
function load_exclusions(dir)
    exclusions = zeros(Int, 0)
    
    # load exclusion file
    files = readdir(dir)
    files = filter(f -> contains(f, "exclusions") && contains(f, ".txt"), files)

    if length(files) == 0
        @warn "No exclusion file found"
        return exclusions
    end
    
    to_int(str) = tryparse(Int, str)
    is_int(str) = to_int(str) !== nothing

    # parse exclusion file
    lines = readlines(dir * "/" * files[1])
    for line in lines
        i = 1
        maxindex = length(line)

        # iterate char by char
        while i <= maxindex
            char = line[i:i]

            # parse number
            if is_int(char)

                num = char
                j = i+1
                while j <= maxindex && is_int(line[j:j])
                    num = num * line[j:j]
                    j += 1
                end

                push!(exclusions, to_int(num))
                i = j

            # parse range
            elseif char == "-"
                if length(exclusions) == 0
                    @warn "Range without start, ignoring"
                    i += 1
                else
                    # sanity check: check that there are numbers on both sides
                    # of the dash
                    j = i-1
                    while j > 0 && line[j:j] == " "
                        j -= 1
                    end
                    before = j > 0 ? line[j:j] : " "
                    
                    i += 1
                    while i <= maxindex && line[i:i] == " "
                        i += 1
                    end
                    after = i <= maxindex ? line[i:i] : " "
                    
                    if !is_int(before) || !is_int(after)
                        @warn "Found dash without 2 numbers around it, ignoring"
                    else
                        # get number
                        num = line[i:i]
                        j = i+1
                        while j <= maxindex && is_int(line[j:j])
                            num = num * line[j:j]
                            j += 1
                        end

                        # get range
                        range_start = exclusions[end]
                        range_end   = to_int(num)
                        range       = collect(range_start:range_end)

                        exclusions = vcat(exclusions[1:end-1], range)
                        i = j
                    end       
                end
                
            # not a number or range: ignore
            else
                i += 1
            end
        end
    end

    return unique(exclusions)
end

"Filename for metadata."
build_fname_meta(fname) = fname * "_meta.csv"

"Filename for time-bins."
build_fname_bins(fname, nbins::Integer, binmode, binalign) =
    fname * "_bins_$(nbins)_$(string(binmode))_$(string(binalign)).csv"


"Process DA signal from raw MATLAB values"
function preprocess_da(raw::Dict; verbose=1)
    # extract time, signal and sampling rate
    x, y = raw["times"], raw["values"]

    interval = raw["interval"]
    fs = 1.0 / interval

    x_start = raw["start"]

    # safety check for sampling rate
    !all(diff(x) .≈ interval) && @error "Missing points from signal"

    # define mapping from time (in seconds) to y index
    yi(t) = round(Int, (t - x_start) * fs + 1)

    # remove outliers > 15 STD away from mean signal
    remove_outliers!(y, 15)

    # compute dF/F signal with 200s moving average
    get_dF_F!(y, fs, 200)

    # smooth with Gaussian window of length 200 ms
    # stddev is taken from default value of MATLAB's gausswin
    # (they use α = 2.5 which translates to σ = 0.5 / α = 0.2)
    win = round(Int, 0.2 * fs)
    y = smooth(y, :gaussian, win, 0.2)

    return x, y, yi
end

"""
    load_preprocess_data(dir[; verbose=1])

Load and preprocess raw data in `dir`.
"""
function load_preprocess_data(dir; verbose=1)
    exclusions = load_exclusions(dir)
    fname, matfile = load_matfile(dir)

    lamp_offs = jdict(matfile[fname * "_Lamp_OFF"])["times"]
    cues      = jdict(matfile[fname * "_Start_Cu"])["times"]
    lamp_ons  = jdict(matfile[fname * "_LampON"  ])["times"]

    licks     = jdict(matfile[fname * "_Lick"])["times"]
    juices    = jdict(matfile[fname * "_Juice"])["times"]

    # safety check
    if length(lamp_offs) > length(cues)
        @warn "More lamp_offs than cues for mouse $fname, trimming..."
        lamp_offs = lamp_offs[1:length(cues)]
    end
    if length(lamp_ons) > length(cues)
        @warn "More lamp_ons than cues for mouse $fname, trimming..."
        lamp_ons = lamp_ons[1:length(cues)]
    end

    # get raw photometry signal
    raw = Dict()
    if fname * "_SNc" in keys(matfile)
        raw = jdict(matfile[fname * "_SNc"])
    elseif fname * "_VTA" in keys(matfile)
        @warn "No SNc measurements found for mouse $fname, using VTA"
        raw = jdict(matfile[fname * "_VTA"])
    else
        @error "No DA measurements found for mouse $fname, skipping"
        return undef, undef
    end

    # get 
    x, y, yi = preprocess_da(raw, verbose=verbose)

    # convert to DataFrame
    ntrials = length(cues)
    meta = DataFrame(
        mouse       = repeat([uppercase(split(fname, "_")[1])], ntrials),
        session     = repeat([string(fname)], ntrials),
        trial       = collect(1:ntrials),
        lamp_off    = lamp_offs,
        cue         = cues,
        lamp_on     = lamp_ons,
        lick        = Vector{Union{Missing, Float64}}(missing, ntrials),
        is_rewarded = falses(ntrials),

        da_baseline = zeros(ntrials),
        da_start    = zeros(ntrials),
        da_end      = zeros(ntrials),
        da_max      = zeros(ntrials),
        da_slope    = zeros(ntrials)
    )

    # remove excluded trials
    valid_trials_idx = filter(i -> !(i in exclusions), meta.trial)
    meta = meta[valid_trials_idx,:]

    for trial in eachrow(meta)
        # find first lick time
        lick_index = findfirst(trial.cue .< licks .< trial.cue + TRIALLEN)

        # handle missing lick
        if lick_index === nothing
            if verbose > 0
                if any(trial.cue .< juices .< trial.lamp_on)
                    lg("No lick for cue $(trial.trial), but reward between cue and lamp on")
                else
                    lg("No lick for cue $(trial.trial)")
                end
            end
            continue
        end

        trial.lick = licks[lick_index]
        trial.is_rewarded = any(trial.lick .<= juices .< trial.lamp_on)
        
        if verbose > 0
            # safety checks
            is_rewarded_3_3 = trial.cue + 3.3 <= trial.lick <= trial.lamp_on
            is_rewarded_5   = trial.cue + 5.0 <= trial.lick <= trial.lamp_on

            if trial.is_rewarded && !is_rewarded_3_3
                lg("Juice delivered, but lick was wrong for trial $(trial.trial)")
            end

            if !trial.is_rewarded && is_rewarded_5
                lg("No juice delivered, but lick was right (> 5s and < 7s) for trial $(trial.trial)")
            end
        end
    end

    # ignore trials where cues happen before lamp off
    meta = meta[findall(meta.lamp_off .< meta.cue), :]

    # ignore trials without licks
    meta = meta[completecases(meta), :]
    disallowmissing!(meta)

    verbose > 0 && lg("Preprocessing $ntrials trials...")
    for trial in eachrow(meta)   
        # compute DA baseline
        trial.da_baseline = mean(y[yi(trial.lamp_off):yi(trial.cue)])        
        
        # compute DA ramp slope
        if trial.lick - trial.cue < BUFFERLEN
            # lick happened very early, so we can't compute anything
            trial.da_start  = trial.da_baseline
            trial.da_end    = trial.da_baseline
            trial.da_max    = trial.da_baseline
        else
            # get DA ramp bounds
            idx1 = yi(trial.cue + BUFFER_POST_CUE)
            idx2 = yi(trial.lick - BUFFER_PRE_LICK)
            
            trial.da_start = y[idx1]
            trial.da_end   = y[idx2]
            trial.da_max   = y[idx1:idx2] |> maximum

            # get slope
            ols = lm(@formula(y ~ x), DataFrame(x=x[idx1:idx2], y=y[idx1:idx2]))
            trial.da_slope = coef(ols)[2] # 2nd coefficient is slope
        end
    end

    verbose > 0 && println("")
    return meta, x, y, yi
end

"""
    get_bins(dir, [nbins::Integer=20;
        binmode::BinMode=AbsBin(),
        binalign::BinAlign=AlignLick(),
        cachedir::String=nothing,
        verbose=1])

Load data from `dir` and divide trials into `nbins` bins.
"""
function get_bins(dir, nbins::Integer=20;
    binmode::BinMode=AbsBin(),
    binalign::BinAlign=AlignLick(),
    cachedir::String=nothing,
    verbose=1)

    # get name of MAT file containing raw values
    fname = get_matfile_fname(dir)

    # get name of files in which results will be stored
    fname_meta = build_fname_meta(fname)
    fname_bins = build_fname_bins(fname, nbins, binmode, binalign)

    # use previously processed data if existing
    if cachedir !== nothing
        cache = readdir(cachedir)
        if fname_meta in cache && fname_bins in cache
            verbose > 0 && lg("Loading $(fname) from cache...")

            meta = DataFrame(CSV.File(cachedir * fname_meta; stringtype=String))
            bins = DataFrame(CSV.File(cachedir * fname_bins))
            
            return meta, bins
        else
            
        end
    end

    verbose > 0 && lg("Loading $(fname) from scratch...")

    # load preprocessed data
    meta, x, y, yi = load_preprocess_data(dir; verbose=verbose)

    # create df for time-bins
    bins = DataFrame(zeros(nrow(meta), nbins), :auto)

    verbose > 0 && lg("Processing $(nrow(meta)) trials...")
    for (trial, bin) in zip(eachrow(meta), eachrow(bins))        
        # compute average DA activity for trial for each time-bin
        # bins completely outside the trial are set to 0
        # bins partially inside the trial are kept 
        bintimes = get_bintimes(trial.cue, trial.lick, nbins, binmode, binalign)
        bin .= map(1:nbins) do j
            if bintimes[j+1] < trial.cue || bintimes[j] > trial.cue + TRIALLEN
                return 0
            elseif bintimes[j] < x[1] || bintimes[j+1] > x[end]
                a = max(x[1], bintimes[j])
                b = min(x[end], bintimes[j+1])

                return mean(y[yi(a):yi(b)])
            else
                return mean(y[yi(bintimes[j]):yi(bintimes[j+1])])
            end
        end
    end

    # save processed data to file
    if cachedir !== nothing
        CSV.write(cachedir * fname_meta, meta)
        CSV.write(cachedir * fname_bins, bins)
    end

    verbose > 0 && println("")
    return meta, bins
end

"""
    get_trial_da(dir[; padding=0.5, verbose=1])

Get DA signal for each trial from data in `dir`. Include `padding` seconds of 
extra signal before cue and after lick.
"""
function get_trial_da(dir; padding=0.5, verbose=1)
    # load preprocessed data
    meta, x, y, yi = load_preprocess_data(dir; verbose=verbose)
    
    # 1000 sanmples per second
    maxlen = ceil(Int, 1000 * (MAX_CUE_TO_LAMP_ON + 2*padding)) + 1
    dopa = missings(Union{Missing, Float64}, maxlen, nrow(meta))

    # get DA on each trial
    for (i, trial) in enumerate(eachrow(meta))
        a = yi(trial.cue - padding)
        b = yi(trial.cue + MAX_CUE_TO_LAMP_ON + padding)
        b = min(b, length(y))

        dopa[1:(b-a+1),i] .= y[a:b]
    end

    return meta, dopa
end


"""
    get_bidirectional_fn(meta_all, bins_all, sessions[;
        baseline_normalization=true,
        lickbin=(0,17),
        reverse=false])

Get coefficients and error of linear regression relating DA bins to 
change in DA ramp slope by pooling the data from `sessions`.
"""
function get_bidirectional_fn(meta_all, bins_all, sessions;
    baseline_normalization=true,
    lickbin=(0,17),
    reverse=false)

    # init empty dfs
    df = similar(bins_all, 0)
    df.a = Float64[]

    df_meta = similar(meta_all, 0)

    for session in sessions
        index = meta_all.session .== session
        
        meta = meta_all[index,:] |> copy
        bins = bins_all[index,:] |> copy
        
        if baseline_normalization
            mapcols!(x -> (x - meta.da_baseline) .* .!iszero.(x), bins)
        end
        
        index = findall(lickbin[1] .< meta.lick .- meta.cue .< lickbin[2])
        if length(index) == 0; continue; end
        
        if index[1] == 1; popfirst!(index); end
        if length(index) == 0; continue; end

        if index[end] == nrow(meta); pop!(index); end
        if length(index) == 0; continue; end
        
        index = index[meta.trial[index.+1] .- meta.trial[index] .== 1]
        
        Δslope = diff(meta.da_slope)[index]
        
        if reverse
            Δslope .*= -1
            index .+= 1
        end

        dfsession = bins[index,:]
        dfsession.a = Δslope
        
        append!(df, dfsession)
        append!(df_meta, meta[index,:])
    end

    formula = Term(:a) ~ ConstantTerm(0) + sum(Term.(Symbol.(names(bins_all))))
    ols     = lm(formula, df)
    coefs   = replace(coef(ols),     NaN => 0)
    errors  = replace(stderror(ols), NaN => 0)

    return coefs, errors, df, df_meta
end
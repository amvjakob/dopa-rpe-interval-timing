# Constants
# Written 21 Oct 2021 by AMVJ

const TRIALLEN = 17          # in seconds
const MAX_CUE_TO_LAMP_ON = 7 # in seconds

const BUFFER_POST_CUE = 0.7  # in seconds
const BUFFER_PRE_LICK = 0.6  # in seconds
const BUFFER_MIN_RAMP = 0.1  # in seconds
const BUFFERLEN = BUFFER_POST_CUE + BUFFER_PRE_LICK + BUFFER_MIN_RAMP

const BUFFER_POST_CUE_MS = round(Int, 1000 * BUFFER_POST_CUE)
const BUFFER_PRE_LICK_MS = round(Int, 1000 * BUFFER_PRE_LICK)
const BUFFER_MIN_RAMP_MS = round(Int, 1000 * BUFFER_MIN_RAMP)
const BUFFERLEN_MS = round(Int, 1000 * BUFFERLEN)

# dirs with VTA measurements
const VTA = [
    "b5_day12_hybop0",
    "b5_day16_allop0_file1",
    "b5_day16_allop0_file2",
    "b5_day18_allop0",
    "b6_day8_hybop0", 
    "h6_day12_hypop0",
    "h6_day9_hybop0"
]

# these dirs are excluded from the analysis (training sessions)
const EXCLUDED_DIRS = [
    "H14_signalname_5",
    "H14_signalname_7",
    "H15_signalname_5"
]


"Logging function."
lg(x...) = println(now(), " ", join(x, " ")...); flush(stdout)

"Compute mean with missing values."
missingmean(A; dims) = mapslices(mean ∘ skipmissing, A, dims=dims)

"Smooth vector `x` with given window `win` (in samples)."
function smooth(x::Vector, win::Symbol=:rect, args...)
    w = getfield(DSP.Windows, win)(args...)
    return DSP.filtfilt(w ./ sum(w), [1.0], x)
end
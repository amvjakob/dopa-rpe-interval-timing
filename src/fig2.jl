# Functions to plot Figure 2
# Written 21 Oct 2021 by AMVJ

include("constants.jl")

using Plots, StatsPlots, ImageFiltering, GLM

"Utility to compute the mean of the matrix `mat` along dim 2."
dopamean(mat) = vec(replace(missingmean(mat, dims=[2]), NaN => 0.0))

"Plot rectangle with given width `w` and height `h` at position (`x`, `y`)."
rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

"""
    plot_bidirectional_plasticity_curve(x, coefs, errors, labels[;
        smooth_win=:rect,
        smooth_len=3,

        xlim=(-6, 3),
        ylim=(-0.22, 0.22),

        annotation_margin_x=0.2,
        annotation_margin_y=0.008])

Plot bidirectional plasticity curve.
"""
function plot_bidirectional_plasticity_curve(x, coefs, errors, labels;
        smooth_win=:rect,
        smooth_len=3,

        xlim=(-6, 3),
        ylim=(-0.22, 0.22),
        
        annotation_margin_x=0.2,
        annotation_margin_y=0.008)

    # define smoothing functions
    smooth_arr(arr) = smooth(arr, smooth_win, smooth_len)
    smooth_mat(mat; dims=[1]) = mapslices(smooth_arr, mat; dims=dims);
       
    # first coefs are from individual mice
    p = plot(x, coefs[:,1:end-1] |> smooth_mat, 
        ribbon=errors[:,1:end-1] |> smooth_mat,
        labels=hcat(labels[1:end-1]...),
        linewidth=1.5,
        fillalpha=0.075,
        opacity=0.35,)

    # last coefs are from pooled data
    plot!(p, x, coefs[:,end] |> smooth_arr,
        ribbon=errors[:,end] |> smooth_arr,
        label=labels[end],
        color=:black,
        linewidth=2,
        fillalpha=0.12,)

    # vertical line at lick time
    vline!(p, [0],
        label="",
        color=:black,
        linestyle=:dash,
        opacity=0.2,)

    # horizontal line at x = 0
    hline!(p, [0],
        label="",
        color=:black,
        linestyle=:dash,
        opacity=0.2,)

    # lick annotation
    scatter!(p, [0], [ylim[1] + annotation_margin_y],
        markershape = :utriangle,
        markersize = 5,
        markeralpha = 1,
        markercolor = :black,
        markerstrokewidth = 1,
        markerstrokealpha = 1,
        markerstrokecolor = :black,
        label="")
    annotate!(p, annotation_margin_x, ylim[1] + 1.5 * annotation_margin_y,
        text("Lick", :black, :left, 9))

    # axis labels
    xlabel!(p, "Lick-aligned time (s)")
    ylabel!(p, "Regression coefficient")

    plot!(p, xlim=xlim,
        ylim=ylim,
        grid=:off,
        legend=:outerright,
        legendtitle="Mouse",
        legendtitlefont=font(10, :right),
        legendfontvalign=:top,
        foreground_color_legend=nothing,)

    return p
end

"""
    plot_fig_2_A_B(x, coefs_A, errors_A, coefs_B, errors_B, labels[; 
        kwargs...])

Plot Fig 2A and 2B.
"""
function plot_fig_2_A_B(x, coefs_A, errors_A, coefs_B, errors_B, labels; 
    kwargs...)

    fig2A = plot_bidirectional_plasticity_curve(x, coefs_A, errors_A, labels;
            kwargs...)

    fig2B = plot_bidirectional_plasticity_curve(x, coefs_B, errors_B, labels;
            kwargs...)

    # fig2A: hide legend
    plot!(fig2A, legend=nothing)

    # fig2B: hide y axis ticks and label
    plot!(fig2B, yticks=([-0.2, -0.1, 0, 0.1, 0.2], ["", "", "", "", ""]))
    ylabel!(fig2B, "")

    # add fig titles
    title!(fig2A, "A\n")
    title!(fig2B, "B\n")

    
    titlex = -1.5
    titley = 0.255
    annotate!(fig2A, titlex, titley, text("Bidirectional plasticity function", 
        :black, :center, 9))
    annotate!(fig2B, titlex, titley, text("Control: reverse analysis", 
        :black, :center, 9))

    # merge fig 2A and 2B
    return plot(fig2A, fig2B,
        titlelocation=:left,
        layout=@layout([a{0.43w} b{0.57w}]),
        left_margin=5Plots.mm
    )
end


"""
    plot_fig2C(x, meta, dopa, indices_cue, indices_lick, lickbin, padding[;
        ylim=(-2, 2),
        annotation_margin_x = 0.075,

        ccue = :black,
        click = colorant"#b5b5b5",
        crect = colorant"#f6f6f6"])

Plot Fig 2A and 2B.
"""
function plot_fig2C(x, meta, dopa, indices_cue, indices_lick, lickbin, padding;
    ylim=(-2, 2),
    annotation_margin_x = 0.075,

    ccue = :black,              # colorant"#00b894"
    click = colorant"#b5b5b5",  # colorant"#e17055"
    crect = colorant"#f6f6f6")

    # compute xlim based on lickbin and padding
    xlim = (-lickbin[2] - 0.2, padding)
    annotation_margin_y = (ylim[2] - ylim[1]) / 60

    # utility to smooth signal
    smooth_plot(y) = smooth(y, :rect, 401)    

    # fit slope to mean DA signal on next trial
    lickbintrials = lickbin[1] .< meta.lick .- meta.cue .< lickbin[2]

    ycue  = 100 * dopamean(dopa[:,lickbintrials .& indices_cue])
    ylick = 100 * dopamean(dopa[:,lickbintrials .& indices_lick])

    function dopafit(x, y)
        a = length(y) - round(Int, 1000 * padding + 1000 * mean(lickbin)) + BUFFER_POST_CUE_MS + 1
        b = length(y) - round(Int, 1000 * padding) - BUFFER_PRE_LICK_MS
        return lm(@formula(y ~ x), DataFrame(x=x[a:b], y=y[a:b]))  
    end

    olscue  = dopafit(x, ycue) 
    olslick = dopafit(x, ylick)

    # init plot
    fig2C = plot()

    # buffer rectangles
    params = Dict(:color => crect, :linecolor => nothing, :label => "")
    plot!(rectangle(-BUFFER_POST_CUE, 4, -mean(lickbin) + BUFFER_POST_CUE, ylim[1]+0.01); params...)
    plot!(rectangle(BUFFER_PRE_LICK_MS, 4, -BUFFER_PRE_LICK, ylim[1]+0.01); params...)

    # thin DA signal
    params = Dict(:linewidth => 1, :linestyle => :dash, :label => "")
    plot!(x, ycue |>  smooth_plot, color=ccue; params...)
    plot!(x, ylick |> smooth_plot, color=click; params...)

    # slope
    params = Dict(:linewidth => 2)
    xdf = DataFrame(x=x[-mean(lickbin) + BUFFER_POST_CUE .< x .< -BUFFER_PRE_LICK])
    plot!(xdf.x, predict(olscue, xdf), label="High DA after cue", color=ccue; params...)
    plot!(xdf.x, predict(olslick, xdf), label="High DA after lick", color=click; params...)

    # vlines at cue and lick
    params = Dict(
        :color => :black,
        :opacity => 0.2,
        :linestyle => :dash,
        :label => ""
    )
    vline!([-mean(lickbin), 0]; params...)

    # triangle markers at cue and lick
    params = Dict(
        :label => "",
        :markershape => :utriangle,
        :markersize => 5,
        :markeralpha => 1,
        :markercolor => :black,
        :markerstrokewidth => 1,
        :markerstrokealpha => 1,
        :markerstrokecolor => :black,
    )
    scatter!([-mean(lickbin), 0], [ylim[1] + annotation_margin_y]; params...)

    # text annotations
    annotate!(-mean(lickbin) + annotation_margin_x, ylim[1] + annotation_margin_y + 0.025,
        text("Cue", :black, :left, 9))
    annotate!(-0 + annotation_margin_x, ylim[1] + annotation_margin_y + 0.025,
        text("Lick", :black, :left, 9))

    # scale annotation
    plot!(rectangle(0.4, ylim[2]-ylim[1], -mean(lickbin) - 0.6, ylim[1]+0.1),
        color=:white,
        linecolor=nothing,
        label="")

    plot!(rectangle(0.03, 1, -mean(lickbin) - 0.4, mean(ylim)-0.5),
        color=:black,
        linecolor=nothing,
        label="")

    # final plot
    plot!(
        legend=nothing,
        grid=:off,
        size=(500, 350),
        xlim=xlim,
        ylim=ylim,
        xlabel="Lick-aligned time (s)",
        ylabel="Dopamine, 1% dF/F",
        yaxis=false,
        yticks=nothing,
        left_margin=5Plots.mm,
    )

    return fig2C
end

"Plot figure 2C with alignment to the cue instead of the lick."
function plot_fig2C_align_cue(x, meta, dopa, indices_cue, indices_lick, lickbin)
    lickbintrials = lickbin[1] .< meta.lick .- meta.cue .< lickbin[2]

    # current trial
    plot()
    plot!(x, dopamean(dopa[:,lickbintrials .& indices_cue]),  label="High cue DA")
    plot!(x, dopamean(dopa[:,lickbintrials .& indices_lick]), label="High lick DA")
    xlabel!("Cue-aligned time")

    return plot!(legend=:topleft, xlim=(0, 1))
end

"""
    plot_fig2(fig2AB, fig2C[; kwargs...])

Combine `fig2AB` and `fig2C` to produce `fig2`.
"""
function plot_fig2(fig2AB, fig2C; kwargs...)
    # add space at bottom
    plot!(fig2AB, bottom_margin=5Plots.mm)

    # add fig 2C title
    title!(fig2C, "C")

    # blank plot for spacing
    pblank = plot(legend=false, grid=false, foreground_color_subplot=:white)

    return plot(fig2AB, pblank, fig2C, pblank,
        titlelocation=:left,
        layout=@layout([a; b{0.2w} c{0.6w} d{0.2w}]),
        size=(1000, 700);
        kwargs...)
end

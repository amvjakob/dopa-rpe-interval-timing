# Functions to plot Figure 1
# Written 21 Oct 2021 by AMVJ

using Plots, LaTeXStrings, DataFrames, GLM

"Plot figure 1A."
function plot_fig1A(t, V, δ)
    twin_margins = Dict(
        :left_margin   => 5Plots.mm,
        :right_margin  => 19Plots.mm,
        :top_margin    => 5Plots.mm,
        :bottom_margin => 7Plots.mm
    )

    params = Dict(:label => "", :xlim => (0, maximum(t)), :linewidth=>2,)
    fig1A = plot(t, V,
        ylabel="Value",
        color=:black;
        params..., twin_margins...)

    fig1A_twin = twinx()
    fig1A_twin_color = :grey
    plot!(fig1A_twin, t[1:length(δ)], δ,
        ylabel="RPE",
        yguidefontcolor=fig1A_twin_color,
        yforeground_color_axis=fig1A_twin_color,
        yforeground_color_border=fig1A_twin_color,
        yforeground_color_guide=fig1A_twin_color,
        yforeground_color_text=fig1A_twin_color,
        color=fig1A_twin_color;
        params..., twin_margins...)

    xlabel!("Time")
    
    return fig1A
end

"Plot figure 1B."
function plot_fig1B(t, dh, T, η)
    fig1B = plot()
    vline!([T], linewidth=2, color=:grey, opacity=0.8, linestyle=:dash, label="")

    # plot slope
    idx = findall(t[1:end-1] ./ η .<= T)
    xdf = DataFrame(t=t[idx] ./ η, y=dh[idx])
    yhat = predict(lm(@formula(y ~ t), xdf), xdf)

    plot!(xdf.t, yhat,
        linewidth=2,
        color=:grey,
        label="DA ramp")

    plot!(t[1:end-1] ./ η, dh,
        linewidth=2,
        color=:black,
        label="DA")

    xlabel!("Time")
    ylabel!("DA Response")

    plot!(
        xlim=(0, maximum(t)),
        ylim=(-0.0005, 0.025), yticks=([0, 0.01, 0.02]),
        legendfontsize=7,
        left_margin=10Plots.mm,)
    
    return fig1B
end

"Plot figure 1C."
function plot_fig1C(t, Vdot, T) 
    fig1C = plot()
    params = Dict(:color=>:grey, :opacity=>.8, :linestyle=>:dash, :label=>"", 
        :linewidth=>2,)
    vline!([T]; params...)
    hline!([0]; params...)

    plot!(t[1:end-1], Vdot,
        xlim=(0, maximum(t)),
        ylim=1.1 .* (minimum(Vdot), maximum(Vdot)),
        color=:black,
        linewidth=2,
        label="",
        left_margin=10Plots.mm,
        right_margin=5Plots.mm,)

    xlabel!("Time")
    ylabel!(L"\partial\hat{V}/\partial\tau")

    return fig1C
end

"Combine `fig1A`, `fig1B` and `fig1C` to create the final figure 1."
function plot_fig1(fig1A, fig1B, fig1C)
    title!(fig1A, "A")
    title!(fig1B, "B")
    title!(fig1C, "C")

    return plot(fig1A, fig1B, fig1C,
        size=(1100,250),
        grid=:off,
        titleloc=:left,
        titlefont=font(16, :bold),
        layout=@layout [a b c])
end
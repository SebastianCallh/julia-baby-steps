module LogisticRegression2
using Revise
using Flux
using Flux: binarycrossentropy, throttle, @epochs
using Flux: Descent, TrackedArray, param, data
using Distributions
using LinearAlgebra
using Random
using Plots
using Base.Iterators: repeated

theme(:juno)

# Generate data
D = 2;   # Data dimension
n = 100; # Number of observations
x1 = rand(MvNormal([1., -5.], I), n);
x2 = rand(MvNormal([-2., -1.5], I), n);
y1 = repeat([1], n);
y2 = repeat([0], n);

classOneColor = :orange
classTwoColor = RGBA(0,.2,.7,1);
p = scatter(x1[1, :], x1[2, :], label="Class 1", color=classOneColor)
scatter!(p, x2[1, :], x2[2, :], label="Class 2", color=classTwoColor)
X = transpose([x1 x2]);
y = [y1; y2];

# Create model
struct LogisticRegression
    α :: TrackedArray # bias
    β :: TrackedArray # weights
    LogisticRegression(α :: Array, β :: Array) = new(param(α), param(β))
end

function (m :: LogisticRegression)(X)
    σ.(m.α .+ X*m.β)
end


function plot_decision_surface(m, xlim :: Tuple{Int, Int}, ylim :: Tuple{Int, Int})
    x = xlim[1]:0.5:xlim[2]
    y = ylim[1]:0.5:ylim[2]
    f = (x, y) -> (first ∘ data ∘ m)([x y])
    contour(x, y, f, fill = true, color = :pu_or)
end


function plot_fit(m)
    xlim = ylim = (-8, 8)
    p = plot_decision_surface(m, xlim, ylim)
    scatter!(p, x1[1, :], x1[2, :], label="Class 1", color=classOneColor)
    scatter!(p, x2[1, :], x2[2, :], label="Class 2", color=classTwoColor)
    p
end

function plot_loss(losses)
    plot(1:length(losses), losses)
end


function create_animation(αs :: Array, βs :: Array)
    anim = @animate for (α, β) in zip(αs, βs)
        m = LogisticRegression(α, β)
        display(plot_fit(m))
    end

    gif(anim, "plots/transition.gif", fps = 60)
end


m = LogisticRegression(rand(1), rand(D, 1)); display(plot_fit(m));
θ = Flux.params(m.α, m.β);
numEpochs = 100;
loss = (x, y) -> mean(binarycrossentropy.(m(x), y));

losses, αs, βs = [], [], []
cb = () -> begin
    l = Flux.data(loss(X, y));
    push!(losses, l);
    push!(αs, Flux.data(m.α));
    push!(βs, Flux.data(m.β));
end

dataset = repeated((X, y), numEpochs)
Flux.train!(loss, θ, dataset, Descent(0.1), cb = cb)

plot_loss(losses)
plot_fit(m)

create_animation(αs, βs)

end

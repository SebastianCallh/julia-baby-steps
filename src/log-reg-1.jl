module LogisticRegression1

using Plots
using Distributions
using LinearAlgebra


# Generate dummy data
n = 100;
x1 = transpose(rand(MvNormal([1., 2.], I), n));
x2 = transpose(rand(MvNormal([-2., -1.5], I), n));
y1 = repeat([1], n);
y2 = repeat([-1], n);
p = scatter(x1[:, 1], x1[:, 2], color = :blue)
scatter!(p, x2[:, 1], x2[:, 2], color = :red)

using Flux
using Flux.Tracker: gradient, update!, data
using Random
using Formatting: printfmt

# Logistic regression
Random.seed!(666);
α = param(rand());   # Bias
β = param(rand(2, 1));  # Weight
predict(X) = σ.(α .+ X*β)
X = [x1; x2];
y = [y1; y2];

function plot_contour(xlim, ylim, f)
    x = xlim[1]:0.5:xlim[2]
    y = ylim[1]:0.5:ylim[2]
    X = repeat(reshape(x, 1, :), length(y), 1)
    Y = repeat(y, 1, length(x))
    contour(x, y, f, fill=true, color=:bluesreds)
    scatter!(x1[:, 1], x1[:, 2],
            color = :red, label="Class 1")
    scatter!(x2[:, 1], x2[:, 2],
            color = :blue, label="Class 2")
end

xlim = ylim = (-5, 5)``
plot_contour(xlim, ylim,
                 (x, y) -> (first ∘ data ∘ predict)([x y]))


function loss(x, y)
    ŷ = predict(x)
    mean(Flux.binarycrossentropy.(ŷ, y))
end

printfmt("Initial loss: {:.4f}", (data ∘ loss)(X, y));

η = 0.01 # learning rate
epochs = 200
plot_interval = 10
anim = @animate for i=1:epochs
    for i in 1:plot_interval
        gs = gradient(() -> loss(X, y), Flux.params(α, β))
        update!(β, -η*gs[β])
        #update!(α, -η*gs[α])
    end
    display(plot_contour(xlim, ylim, (x, y) ->
                 (first ∘ data ∘ predict)([x y])))
end
gif(anim, "bluesreds.gif", fps = 60)

printfmt("Loss after training: {:.4f}", (data ∘ loss)(X, y));

end

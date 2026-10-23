"""Scale `s` in `ΔW = s·B·A`."""
abstract type LoRAScaling end

"""`s = α/r` (Hu et al. 2021, arXiv:2106.09685 §4.1)."""
struct StandardScaling <: LoRAScaling end

"""`s = α/√r` (rsLoRA, Kalajdzievski 2023, arXiv:2312.03732)."""
struct RankStabilizedScaling <: LoRAScaling end

lora_scale(::StandardScaling, α::Real, r::Integer) = α / r
lora_scale(::RankStabilizedScaling, α::Real, r::Integer) = α / sqrt(r)

"""Distribution of a frozen or initial random factor with `fan_in` columns."""
abstract type FactorInit end

"""
    KaimingUniformInit(a)

torch `kaiming_uniform_(·, a)` in `fan_in` mode: `U(-bound, bound)` with `bound = √3·√(2/(1+a²))/√fan_in`
(torch/nn/init.py). `a = √5` is the loralib and PEFT initialization of LoRA's `A` (`bound = 1/√fan_in`);
`a = 0` is PEFT's VeRA `_kaiming_init` (`bound = √(6/fan_in)`).
"""
struct KaimingUniformInit{T <: Real} <: FactorInit
    a::T
end

kaiming_bound(init::KaimingUniformInit, fan_in::Integer) = sqrt(3) * sqrt(2 / (1 + init.a^2)) / sqrt(fan_in)

"""`N(0, std²)` entries."""
struct GaussianInit{T <: Real} <: FactorInit
    std::T
end

draw_factor(rng::Random.AbstractRNG, init::KaimingUniformInit, rows::Integer, fan_in::Integer) =
    kaiming_bound(init, fan_in) .* (2 .* Random.rand(rng, rows, fan_in) .- 1)
draw_factor(rng::Random.AbstractRNG, init::GaussianInit, rows::Integer, fan_in::Integer) =
    init.std .* Random.randn(rng, rows, fan_in)

"""
    LoRA(weights; rank, alpha, scaling, init, rng)

`ΔW = s·B·A` with `A ∈ R^{r×n_in}`, `B ∈ R^{n_out×r}` both trainable (Hu et al. 2021). The reference point is
`A = A₀` drawn from `init`, `B = 0`. `θ` per matrix: `vec(A)` then `vec(B)`, column-major.
"""
struct LoRA{FT <: AbstractFloat, M <: AbstractMatrix{FT}, VN <: AbstractVector{NTuple{2, Int}}, VI <: AbstractVector{<:Integer}, VM <: AbstractVector{M}} <: Adapter
    idx::VI
    shapes::VN
    offsets::VI
    n::Int
    rank::Int
    scale::FT
    A0::VM
end

function LoRA(
    base::AbstractVector{<:AbstractMatrix};
    rank::Integer,
    alpha::Real,
    scaling::LoRAScaling,
    init::FactorInit,
    rng::Random.AbstractRNG,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1"))
    idx, shapes = resolve_arrays(base)
    sizes = [rank * (no + ni) for (no, ni) in shapes]
    offsets = cumsum([0; sizes[1:(end - 1)]])
    A0 = [draw_factor(rng, init, rank, ni) for (_, ni) in shapes]
    return LoRA(idx, shapes, offsets, sum(sizes), Int(rank), float(lora_scale(scaling, alpha, rank)), A0)
end

nparams(a::LoRA) = a.n
reference_point(a::LoRA) =
    reduce(vcat, [vcat(vec(a.A0[k]), zeros(no * a.rank)) for (k, (no, _)) in enumerate(a.shapes)])
segments(a::LoRA) = reduce(
    vcat,
    [
        [:A => (o + 1):(o + a.rank * ni), :B => (o + a.rank * ni + 1):(o + a.rank * (ni + no))] for
        (o, (no, ni)) in zip(a.offsets, a.shapes)
    ],
)

function add_delta!(out::AbstractVector{<:AbstractMatrix}, a::LoRA, base::AbstractVector{<:AbstractMatrix}, θ::AbstractVector)
    check_inputs(a, out, base, θ)
    r, s = a.rank, a.scale
    @inbounds for (k, ix) in enumerate(a.idx)
        W = out[ix]
        no, ni = a.shapes[k]
        oA = a.offsets[k]
        oB = oA + r * ni
        for j in 1:ni, i in 1:no
            acc = zero(eltype(θ))
            for q in 1:r
                acc += θ[oB + i + (q - 1) * no] * θ[oA + q + (j - 1) * r]
            end
            W[i, j] += s * acc
        end
    end
    return out
end

"""
    LoRAFA(weights; rank, alpha, scaling, init, rng)

`ΔW = s·B·A₀` with `A₀` frozen (drawn from `init`) and only `B ∈ R^{n_out×r}` trainable (LoRA-FA, Zhang et al.
2023, arXiv:2308.03303). The reference point is `B = 0`. `θ` per matrix: `vec(B)`, column-major.
"""
struct LoRAFA{FT <: AbstractFloat, M <: AbstractMatrix{FT}, VN <: AbstractVector{NTuple{2, Int}}, VI <: AbstractVector{<:Integer}, VM <: AbstractVector{M}} <: Adapter
    idx::VI
    shapes::VN
    offsets::VI
    n::Int
    rank::Int
    scale::FT
    A0::VM
end

function LoRAFA(
    base::AbstractVector{<:AbstractMatrix};
    rank::Integer,
    alpha::Real,
    scaling::LoRAScaling,
    init::FactorInit,
    rng::Random.AbstractRNG,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1"))
    idx, shapes = resolve_arrays(base)
    sizes = [rank * no for (no, _) in shapes]
    offsets = cumsum([0; sizes[1:(end - 1)]])
    A0 = [draw_factor(rng, init, rank, ni) for (_, ni) in shapes]
    return LoRAFA(idx, shapes, offsets, sum(sizes), Int(rank), float(lora_scale(scaling, alpha, rank)), A0)
end

nparams(a::LoRAFA) = a.n
reference_point(a::LoRAFA) = zeros(a.n)
segments(a::LoRAFA) = [:B => (o + 1):(o + a.rank * no) for (o, (no, _)) in zip(a.offsets, a.shapes)]

function add_delta!(out::AbstractVector{<:AbstractMatrix}, a::LoRAFA, base::AbstractVector{<:AbstractMatrix}, θ::AbstractVector)
    check_inputs(a, out, base, θ)
    r, s = a.rank, a.scale
    @inbounds for (k, ix) in enumerate(a.idx)
        W = out[ix]
        A = a.A0[k]
        no, ni = a.shapes[k]
        o = a.offsets[k]
        for j in 1:ni, i in 1:no
            acc = zero(eltype(θ))
            for q in 1:r
                acc += θ[o + i + (q - 1) * no] * A[q, j]
            end
            W[i, j] += s * acc
        end
    end
    return out
end

"""
    PiSSA(weights; rank, alpha, scaling)

`W = W_res + s·B·A` (PiSSA, Meng et al. 2024, arXiv:2404.02948) with, from the rank-`r` truncated SVD of the
base `W`, `A₀ = diag(√(σ_r/s))·V_rᵀ` and `B₀ = U_r·diag(√(σ_r/s))` as in PEFT's `pissa_init`, so `s·B₀·A₀`
is the principal part and `W_res = W − s·B₀·A₀` is frozen. Both factors are trainable and the reference point
is `(A₀, B₀)`. The change added to `W` is `s·(B·A − B₀·A₀)`. `θ` per matrix: `vec(A)` then `vec(B)`,
column-major.
"""
struct PiSSA{
    FT <: AbstractFloat,
    VN <: AbstractVector{NTuple{2, Int}},
    VI <: AbstractVector{<:Integer},
    VM <: AbstractVector{<:AbstractMatrix{FT}},
} <: Adapter
    idx::VI
    shapes::VN
    offsets::VI
    n::Int
    rank::Int
    scale::FT
    A0::VM
    B0::VM
end

function PiSSA(base::AbstractVector{<:AbstractMatrix}; rank::Integer, alpha::Real, scaling::LoRAScaling)
    idx, shapes = resolve_arrays(base)
    sc = float(lora_scale(scaling, alpha, rank))
    factors = [truncated_svd(base[ix], rank) for ix in idx]
    A0 = [sqrt.(σ ./ sc) .* permutedims(V) for (_, σ, V) in factors]
    B0 = [U .* permutedims(sqrt.(σ ./ sc)) for (U, σ, _) in factors]
    sizes = [rank * (no + ni) for (no, ni) in shapes]
    offsets = cumsum([0; sizes[1:(end - 1)]])
    return PiSSA(idx, shapes, offsets, sum(sizes), Int(rank), sc, A0, B0)
end

nparams(a::PiSSA) = a.n
reference_point(a::PiSSA) = reduce(vcat, [vcat(vec(a.A0[k]), vec(a.B0[k])) for k in eachindex(a.A0)])
segments(a::PiSSA) = reduce(
    vcat,
    [
        [:A => (o + 1):(o + a.rank * ni), :B => (o + a.rank * ni + 1):(o + a.rank * (ni + no))] for
        (o, (no, ni)) in zip(a.offsets, a.shapes)
    ],
)

function add_delta!(out::AbstractVector{<:AbstractMatrix}, a::PiSSA, base::AbstractVector{<:AbstractMatrix}, θ::AbstractVector)
    check_inputs(a, out, base, θ)
    r, s = a.rank, a.scale
    @inbounds for (k, ix) in enumerate(a.idx)
        W = out[ix]
        A0, B0 = a.A0[k], a.B0[k]
        no, ni = a.shapes[k]
        oA = a.offsets[k]
        oB = oA + r * ni
        for j in 1:ni, i in 1:no
            acc = zero(eltype(θ))
            for q in 1:r
                acc += θ[oB + i + (q - 1) * no] * θ[oA + q + (j - 1) * r] - B0[i, q] * A0[q, j]
            end
            W[i, j] += s * acc
        end
    end
    return out
end

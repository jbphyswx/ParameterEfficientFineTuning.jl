"""
    LoRAXS(weights; rank, alpha, scaling)

`ΔW = s·U_r·R·Σ_r·V_rᵀ` with `U_r, Σ_r, V_r` the frozen rank-`r` truncated SVD of the base `W` and only
`R ∈ R^{r×r}` trainable (LoRA-XS, Bałazy et al. 2024, arXiv:2405.17604, in the order of the reference
implementation, `Σ_r` on the input side of `R`). The reference point is `R = 0`. `θ` per matrix: `vec(R)`,
column-major.
"""
struct LoRAXS{
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
    U::VM
    Q::VM
end

function LoRAXS(base::AbstractVector{<:AbstractMatrix}; rank::Integer, alpha::Real, scaling::LoRAScaling)
    idx, shapes = resolve_arrays(base)
    factors = [truncated_svd(base[ix], rank) for ix in idx]
    U = [f[1] for f in factors]
    Q = [f[2] .* permutedims(f[3]) for f in factors]
    sizes = fill(rank^2, length(idx))
    offsets = cumsum([0; sizes[1:(end - 1)]])
    return LoRAXS(idx, shapes, offsets, sum(sizes), Int(rank), float(lora_scale(scaling, alpha, rank)), U, Q)
end

nparams(a::LoRAXS) = a.n
reference_point(a::LoRAXS) = zeros(a.n)
segments(a::LoRAXS) = [:R => (o + 1):(o + a.rank^2) for o in a.offsets]

function add_delta!(out::AbstractVector{<:AbstractMatrix}, a::LoRAXS, base::AbstractVector{<:AbstractMatrix}, θ::AbstractVector)
    check_inputs(a, out, base, θ)
    r, s = a.rank, a.scale
    @inbounds for (k, ix) in enumerate(a.idx)
        W = out[ix]
        U, Q = a.U[k], a.Q[k]
        no, ni = a.shapes[k]
        o = a.offsets[k]
        for j in 1:ni, p in 1:r
            t = zero(eltype(θ))
            for q in 1:r
                t += θ[o + p + (q - 1) * r] * Q[q, j]
            end
            t *= s
            for i in 1:no
                W[i, j] += U[i, p] * t
            end
        end
    end
    return out
end

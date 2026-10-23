"""
    TinyLoRA(weights, groups; rank, u, projection_init, rng)

`ΔW = U_r·Σ_r·(Σ_k v_k P_k)·V_rᵀ` (TinyLoRA, Morris et al. 2026, arXiv:2602.04118) with `U_r, Σ_r, V_r` the
frozen rank-`r` truncated SVD of the base `W`, fixed random `P_k ∈ R^{r×r}` (`k = 1:u`) per matrix drawn from
`projection_init`, and one trainable `v ∈ R^u` shared by every matrix of a group (the paper's weight tying). The
reference point is `v = 0`. `groups` partitions the indices of `weights`. `θ`: each group's `v`, in the order of `groups`.
"""
struct TinyLoRA{
    FT <: AbstractFloat,
    VN <: AbstractVector{NTuple{2, Int}},
    VI <: AbstractVector{<:Integer},
    VG <: AbstractVector{<:Integer},
    VM <: AbstractVector{<:AbstractMatrix{FT}},
    VP <: AbstractVector{<:AbstractArray{FT, 3}},
} <: Adapter
    idx::VI
    shapes::VN
    group::VG
    ngroups::Int
    u::Int
    rank::Int
    US::VM
    Vt::VM
    P::VP
end

function TinyLoRA(
    base::AbstractVector{<:AbstractMatrix},
    groups::AbstractVector{<:AbstractVector{<:Integer}};
    rank::Integer,
    u::Integer,
    projection_init::FactorInit,
    rng::Random.AbstractRNG,
)
    u >= 1 || throw(ArgumentError("u must be ≥ 1"))
    isempty(groups) && throw(ArgumentError("TinyLoRA needs at least one group"))
    any(isempty, groups) && throw(ArgumentError("TinyLoRA groups must be non-empty"))
    idx = collect(Int, reduce(vcat, groups))
    group = reduce(vcat, [fill(g, length(sg)) for (g, sg) in enumerate(groups)])
    resolve_arrays(base)
    sort(idx) == collect(eachindex(base)) ||
        throw(ArgumentError("TinyLoRA groups must partition the weight indices"))
    shapes = [size(base[i]) for i in idx]
    factors = [truncated_svd(base[ix], rank) for ix in idx]
    US = [f[1] .* permutedims(f[2]) for f in factors]
    Vt = [permutedims(f[3]) for f in factors]
    P = [cat((draw_factor(rng, projection_init, rank, rank) for _ in 1:u)...; dims = 3) for _ in idx]
    return TinyLoRA(idx, shapes, group, length(groups), Int(u), Int(rank), US, Vt, P)
end

nparams(a::TinyLoRA) = a.u * a.ngroups
reference_point(a::TinyLoRA) = zeros(nparams(a))
segments(a::TinyLoRA) = [:v => ((g - 1) * a.u + 1):(g * a.u) for g in 1:a.ngroups]

function add_delta!(out::AbstractVector{<:AbstractMatrix}, a::TinyLoRA, base::AbstractVector{<:AbstractMatrix}, θ::AbstractVector)
    check_inputs(a, out, base, θ)
    r, u = a.rank, a.u
    @inbounds for (k, ix) in enumerate(a.idx)
        W = out[ix]
        US, Vt, P = a.US[k], a.Vt[k], a.P[k]
        no, ni = a.shapes[k]
        o = (a.group[k] - 1) * u
        for j in 1:ni, p in 1:r
            t = zero(eltype(θ))
            for q in 1:r
                Rpq = zero(eltype(θ))
                for m in 1:u
                    Rpq += θ[o + m] * P[p, q, m]
                end
                t += Rpq * Vt[q, j]
            end
            for i in 1:no
                W[i, j] += US[i, p] * t
            end
        end
    end
    return out
end

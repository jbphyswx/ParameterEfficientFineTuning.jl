"""
    RandomSubspace(arrays; dim, rng)

`Δ = P·θ` over the concatenated, column-major entries of the supplied arrays, with
`P ∈ R^{D×dim}` a frozen Gaussian matrix whose columns are normalized to unit length
(Li et al. 2018, arXiv:1804.08838, Eq. 2). The reference point is `θ = 0`.
"""
struct RandomSubspace{FT <: AbstractFloat, M <: AbstractMatrix{FT}, VS <: AbstractVector{<:Tuple}, VI <: AbstractVector{<:Integer}} <: Adapter
    idx::VI
    shapes::VS
    offsets::VI
    D::Int
    Pt::M
end

function RandomSubspace(base::AbstractVector{<:AbstractArray}; dim::Integer, rng::Random.AbstractRNG)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))
    idx, shapes = resolve_arrays(base)
    sizes = map(length, base)
    offsets = cumsum([0; sizes[1:(end - 1)]])
    D = sum(sizes)
    P = Random.randn(rng, D, dim)
    for j in 1:dim
        P[:, j] ./= sqrt(sum(abs2, view(P, :, j)))
    end
    return RandomSubspace(idx, shapes, offsets, D, permutedims(P))
end

nparams(a::RandomSubspace) = size(a.Pt, 1)
reference_point(a::RandomSubspace) = zeros(nparams(a))
segments(a::RandomSubspace) = [:coefficients => 1:nparams(a)]

"""The `D × dim` projection `P`."""
projection(a::RandomSubspace) = permutedims(a.Pt)

@inline function _project(Pt::AbstractMatrix, p::Integer, θ::AbstractVector)
    acc = zero(promote_type(eltype(Pt), eltype(θ)))
    @inbounds for j in axes(Pt, 1)
        acc += Pt[j, p] * θ[j]
    end
    return acc
end

function add_delta!(out::AbstractVector{<:AbstractArray}, a::RandomSubspace, base::AbstractVector{<:AbstractArray}, θ::AbstractVector)
    check_inputs(a, out, base, θ)
    Pt = a.Pt
    @inbounds for (k, ix) in enumerate(a.idx)
        array = out[ix]
        o = a.offsets[k]
        for (j, i) in enumerate(eachindex(array))
            array[i] += _project(Pt, o + j, θ)
        end
    end
    return out
end

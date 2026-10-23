"""
A parameterization `θ ↦ Δ` of changes to an ordered collection of parameter arrays.

Adapters implement [`nparams`](@ref), [`reference_point`](@ref), [`segments`](@ref), and
[`add_delta!`](@ref). An adapter may be shared by concurrent tasks with separate output arrays.
The caller selects the arrays; adapters have no knowledge of the model containing them.
"""
abstract type Adapter end

"""Number of trainable parameters of an adapter."""
function nparams end

"""`θ₀` at which the adapter changes nothing."""
function reference_point end

"""
    segments(adapter) -> Vector{Pair{Symbol, UnitRange{Int}}}

Named pieces of `θ`, covering `1:nparams(adapter)` in order.
"""
function segments end

"""
    add_delta!(out, adapter, base, θ)

Add the adapter's change to the ordered arrays in `out`. `base` supplies frozen parameters
for adapters whose change depends on them. Both collections must match the construction shapes.
"""
function add_delta! end

"""
    apply!(out, adapter, base, θ)

Reset the selected arrays in `out` to `base`, then add the adapter's change at `θ`.
The output arrays must not alias the frozen base arrays.
"""
function apply!(out::AbstractVector{<:AbstractArray}, a::Adapter, base::AbstractVector{<:AbstractArray}, θ::AbstractVector)
    check_inputs(a, out, base, θ)
    for i in eachindex(base)
        copyto!(out[i], base[i])
    end
    add_delta!(out, a, base, θ)
    return out
end

function resolve_arrays(base::AbstractVector{<:AbstractArray})
    isempty(base) && throw(ArgumentError("parameter selection is empty"))
    Base.require_one_based_indexing(base)
    for array in base
        Base.require_one_based_indexing(array)
    end
    return collect(eachindex(base)), map(size, base)
end

function check_arrays(a::Adapter, arrays::AbstractVector{<:AbstractArray})
    Base.require_one_based_indexing(arrays)
    length(arrays) == length(a.shapes) || throw(DimensionMismatch("parameter array counts differ"))
    for (k, i) in enumerate(a.idx)
        Base.require_one_based_indexing(arrays[i])
        size(arrays[i]) == a.shapes[k] || throw(DimensionMismatch("parameter array $i has the wrong shape"))
    end
    return nothing
end

function check_inputs(a::Adapter, out, base, θ::AbstractVector)
    Base.require_one_based_indexing(θ)
    length(θ) == nparams(a) || throw(DimensionMismatch("adapter has $(nparams(a)) parameters, θ has $(length(θ))"))
    check_arrays(a, out)
    check_arrays(a, base)
    return nothing
end

"""Rank-`r` truncated SVD of `W` in `Float64`: `(U_r, σ_r, V_r)` with `W ≈ U_r * Diagonal(σ_r) * V_r'`."""
function truncated_svd(W::AbstractMatrix, r::Integer)
    1 <= r <= minimum(size(W)) || throw(ArgumentError("rank $r is outside 1:$(minimum(size(W))) for a $(size(W)) weight"))
    F = LinearAlgebra.svd(Float64.(W))
    return F.U[:, 1:r], F.S[1:r], F.V[:, 1:r]
end

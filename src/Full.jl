"""
    Full(arrays; names = fill(:W, length(arrays)))

Free additive changes to every supplied parameter array. `θ` concatenates their entries in
column-major order. `names` labels the segments; for weights and biases supplied alternately,
use `[:W, :b, …]` to retain separate weight and bias prior scales.
"""
struct Full{VS <: AbstractVector{<:Tuple}, VI <: AbstractVector{<:Integer}} <: Adapter
    idx::VI
    shapes::VS
    offsets::VI
    n::Int
    names::Vector{Symbol}
end

function Full(base::AbstractVector{<:AbstractArray}; names::AbstractVector{Symbol} = fill(:W, length(base)))
    idx, shapes = resolve_arrays(base)
    length(names) == length(base) || throw(DimensionMismatch("one segment name is needed per parameter array"))
    sizes = map(length, base)
    offsets = cumsum([0; sizes[1:(end - 1)]])
    return Full(idx, shapes, offsets, sum(sizes), collect(names))
end

nparams(a::Full) = a.n
reference_point(a::Full) = zeros(a.n)
segments(a::Full) = [name => ((o + 1):(o + prod(shape))) for (name, o, shape) in zip(a.names, a.offsets, a.shapes)]

function add_delta!(out::AbstractVector{<:AbstractArray}, a::Full, base::AbstractVector{<:AbstractArray}, θ::AbstractVector)
    check_inputs(a, out, base, θ)
    @inbounds for (k, ix) in enumerate(a.idx)
        array = out[ix]
        o = a.offsets[k]
        for (j, i) in enumerate(eachindex(array))
            array[i] += θ[o + j]
        end
    end
    return out
end

"""
    BitFit(biases)

Additive bias updates (Ben Zaken et al. 2022, arXiv:2106.10199). The caller supplies only
bias arrays. `θ` concatenates their entries in column-major order.
"""
struct BitFit{VS <: AbstractVector{<:Tuple}, VI <: AbstractVector{<:Integer}} <: Adapter
    idx::VI
    shapes::VS
    offsets::VI
    n::Int
end

function BitFit(base::AbstractVector{<:AbstractArray})
    idx, shapes = resolve_arrays(base)
    sizes = map(length, base)
    offsets = cumsum([0; sizes[1:(end - 1)]])
    return BitFit(idx, shapes, offsets, sum(sizes))
end

nparams(a::BitFit) = a.n
reference_point(a::BitFit) = zeros(a.n)
segments(a::BitFit) = [:b => ((o + 1):(o + prod(shape))) for (o, shape) in zip(a.offsets, a.shapes)]

function add_delta!(out::AbstractVector{<:AbstractArray}, a::BitFit, base::AbstractVector{<:AbstractArray}, θ::AbstractVector)
    check_inputs(a, out, base, θ)
    @inbounds for (k, ix) in enumerate(a.idx)
        bias = out[ix]
        o = a.offsets[k]
        for (j, i) in enumerate(eachindex(bias))
            bias[i] += θ[o + j]
        end
    end
    return out
end

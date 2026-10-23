"""
    FourierFT(weights; n_frequency, alpha, location_seed)

`ΔW = α·Re(ifft2(F))` (FourierFT, Gao et al. 2024, arXiv:2405.03003, as in PEFT's `get_delta_weight`), where the
`(n_out, n_in)` spectrum `F` is zero except at `n_frequency` locations holding the trainable coefficients `c`.
Locations are the first `n_frequency` entries of a random permutation of `0:(n_out·n_in − 1)` seeded with
`location_seed` for every matrix, read row-major (`row = idx ÷ n_in`, `col = idx % n_in`), so same-shaped matrices
share locations. With the `ifft2` normalization, `ΔW[p,q] = α/(n_out·n_in)·Σ_m c_m·cos(2π(p·j_m/n_out + q·k_m/n_in))`
for 0-based `p, q, j_m, k_m`. The reference point is `c = 0`. `θ` per matrix: `c`.
"""
struct FourierFT{
    FT <: AbstractFloat,
    VN <: AbstractVector{NTuple{2, Int}},
    VI <: AbstractVector{<:Integer},
    VL <: AbstractVector{<:AbstractMatrix{<:Integer}},
} <: Adapter
    idx::VI
    shapes::VN
    offsets::VI
    n::Int
    n_frequency::Int
    alpha::FT
    locations::VL
end

function FourierFT(base::AbstractVector{<:AbstractMatrix}; n_frequency::Integer, alpha::Real, location_seed::Integer)
    idx, shapes = resolve_arrays(base)
    for (no, ni) in shapes
        1 <= n_frequency <= no * ni ||
            throw(ArgumentError("n_frequency $n_frequency is outside 1:$(no * ni) for a ($no, $ni) weight"))
    end
    locations = map(shapes) do (no, ni)
        flat = Random.randperm(Random.MersenneTwister(location_seed), no * ni)[1:n_frequency] .- 1
        permutedims(hcat(flat .÷ ni, flat .% ni))
    end
    sizes = fill(Int(n_frequency), length(idx))
    offsets = cumsum([0; sizes[1:(end - 1)]])
    return FourierFT(idx, shapes, offsets, sum(sizes), Int(n_frequency), float(alpha), locations)
end

nparams(a::FourierFT) = a.n
reference_point(a::FourierFT) = zeros(a.n)
segments(a::FourierFT) = [:c => (o + 1):(o + a.n_frequency) for o in a.offsets]

"""The `2 × n_frequency` 0-based (row, column) spectrum locations of each site."""
locations(a::FourierFT) = a.locations

function add_delta!(out::AbstractVector{<:AbstractMatrix}, a::FourierFT, base::AbstractVector{<:AbstractMatrix}, θ::AbstractVector)
    check_inputs(a, out, base, θ)
    @inbounds for (k, ix) in enumerate(a.idx)
        W = out[ix]
        L = a.locations[k]
        no, ni = a.shapes[k]
        o = a.offsets[k]
        norm = a.alpha / (no * ni)
        for q in 1:ni, p in 1:no
            acc = zero(eltype(θ))
            for m in 1:a.n_frequency
                acc += θ[o + m] * cospi(2 * ((p - 1) * L[1, m] / no + (q - 1) * L[2, m] / ni))
            end
            W[p, q] += norm * acc
        end
    end
    return out
end

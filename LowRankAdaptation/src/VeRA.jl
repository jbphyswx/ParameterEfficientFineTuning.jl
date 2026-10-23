"""
    VeRA(weights; rank, d_init, init, rng)

`ΔW = Λ_b·B·Λ_d·A` (VeRA, Kopiczko et al. 2024, arXiv:2310.11454) with `A ∈ R^{r×n_in,max}` and
`B ∈ R^{n_out,max×r}` frozen, drawn once from `init` (fan-in `n_in,max` and `r`), shared by every matrix and sliced
to `A[:, 1:n_in]`, `B[1:n_out, :]` as in PEFT, and trainable `b ∈ R^{n_out}`, `d ∈ R^r` per matrix. The reference
point is `b = 0`, `d = d_init`, so `∂ΔW/∂b ≠ 0` there. `θ` per matrix: `b` then `d`.
"""
struct VeRA{
    FT <: AbstractFloat,
    VN <: AbstractVector{NTuple{2, Int}},
    VI <: AbstractVector{<:Integer},
    M <: AbstractMatrix{FT},
} <: Adapter
    idx::VI
    shapes::VN
    offsets::VI
    n::Int
    rank::Int
    d_init::FT
    A::M
    B::M
end

function VeRA(
    base::AbstractVector{<:AbstractMatrix};
    rank::Integer,
    d_init::Real,
    init::FactorInit,
    rng::Random.AbstractRNG,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1"))
    idx, shapes = resolve_arrays(base)
    no_max, ni_max = maximum(first, shapes), maximum(last, shapes)
    A = draw_factor(rng, init, rank, ni_max)
    B = draw_factor(rng, init, no_max, rank)
    sizes = [no + rank for (no, _) in shapes]
    offsets = cumsum([0; sizes[1:(end - 1)]])
    return VeRA(idx, shapes, offsets, sum(sizes), Int(rank), float(d_init), A, B)
end

nparams(a::VeRA) = a.n
reference_point(a::VeRA) = reduce(vcat, [vcat(zeros(no), fill(a.d_init, a.rank)) for (no, _) in a.shapes])
segments(a::VeRA) =
    reduce(vcat, [[:b => (o + 1):(o + no), :d => (o + no + 1):(o + no + a.rank)] for (o, (no, _)) in zip(a.offsets, a.shapes)])

function add_delta!(out::AbstractVector{<:AbstractMatrix}, a::VeRA, base::AbstractVector{<:AbstractMatrix}, θ::AbstractVector)
    check_inputs(a, out, base, θ)
    r, A, B = a.rank, a.A, a.B
    @inbounds for (k, ix) in enumerate(a.idx)
        W = out[ix]
        no, ni = a.shapes[k]
        ob = a.offsets[k]
        od = ob + no
        for j in 1:ni, i in 1:no
            acc = zero(eltype(θ))
            for q in 1:r
                acc += B[i, q] * θ[od + q] * A[q, j]
            end
            W[i, j] += θ[ob + i] * acc
        end
    end
    return out
end

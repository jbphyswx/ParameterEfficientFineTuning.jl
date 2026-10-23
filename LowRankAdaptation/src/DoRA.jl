"""
    DoRA(weights; rank, alpha, scaling, init, rng)

`W' = m ⊙ (W₀ + s·B·A) / ‖W₀ + s·B·A‖_row` (DoRA, Liu et al. 2024, arXiv:2402.09353), the norm taken over each
output row of the `(n_out, n_in)` weight as in PEFT (`linalg.norm(weight, dim=1)`), with `m ∈ R^{n_out}`,
`A ∈ R^{r×n_in}`, `B ∈ R^{n_out×r}` trainable. The reference point is `m = ‖W₀‖_row`, `A = A₀` drawn from `init`,
`B = 0`. The change added to `W` is `W' − W₀`, which depends on the base weights. `θ` per matrix: `m`, `vec(A)`,
`vec(B)`, column-major.
"""
struct DoRA{
    FT <: AbstractFloat,
    VN <: AbstractVector{NTuple{2, Int}},
    VI <: AbstractVector{<:Integer},
    VM <: AbstractVector{<:AbstractMatrix{FT}},
    VV <: AbstractVector{<:AbstractVector{FT}},
} <: Adapter
    idx::VI
    shapes::VN
    offsets::VI
    n::Int
    rank::Int
    scale::FT
    A0::VM
    m0::VV
end

function _row_norm(W0::AbstractMatrix, i::Integer)
    acc = 0.0
    @inbounds for j in axes(W0, 2)
        v = Float64(W0[i, j])
        acc += v * v
    end
    return sqrt(acc)
end

function DoRA(
    base::AbstractVector{<:AbstractMatrix};
    rank::Integer,
    alpha::Real,
    scaling::LoRAScaling,
    init::FactorInit,
    rng::Random.AbstractRNG,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1"))
    idx, shapes = resolve_arrays(base)
    A0 = [draw_factor(rng, init, rank, ni) for (_, ni) in shapes]
    m0 = [[_row_norm(base[ix], i) for i in 1:shapes[k][1]] for (k, ix) in enumerate(idx)]
    any(any(iszero, m) for m in m0) && throw(ArgumentError("DoRA needs every base row to have nonzero norm"))
    sizes = [no + rank * (no + ni) for (no, ni) in shapes]
    offsets = cumsum([0; sizes[1:(end - 1)]])
    return DoRA(idx, shapes, offsets, sum(sizes), Int(rank), float(lora_scale(scaling, alpha, rank)), A0, m0)
end

nparams(a::DoRA) = a.n
reference_point(a::DoRA) =
    reduce(vcat, [vcat(a.m0[k], vec(a.A0[k]), zeros(no * a.rank)) for (k, (no, _)) in enumerate(a.shapes)])
segments(a::DoRA) = reduce(
    vcat,
    [
        [
            :m => (o + 1):(o + no),
            :A => (o + no + 1):(o + no + a.rank * ni),
            :B => (o + no + a.rank * ni + 1):(o + no + a.rank * (ni + no)),
        ] for (o, (no, ni)) in zip(a.offsets, a.shapes)
    ],
)

@inline function _dora_entry(W0, θ, i, j, oA, oB, no, r, s)
    acc = zero(eltype(θ))
    @inbounds for q in 1:r
        acc += θ[oB + i + (q - 1) * no] * θ[oA + q + (j - 1) * r]
    end
    return Float64(@inbounds W0[i, j]) + s * acc
end

function add_delta!(out::AbstractVector{<:AbstractMatrix}, a::DoRA, base::AbstractVector{<:AbstractMatrix}, θ::AbstractVector)
    check_inputs(a, out, base, θ)
    r, s = a.rank, a.scale
    @inbounds for (k, ix) in enumerate(a.idx)
        W = out[ix]
        W0 = base[ix]
        no, ni = a.shapes[k]
        om = a.offsets[k]
        oA = om + no
        oB = oA + r * ni
        for i in 1:no
            nrm2 = 0.0
            for j in 1:ni
                v = _dora_entry(W0, θ, i, j, oA, oB, no, r, s)
                nrm2 += v * v
            end
            c = θ[om + i] / sqrt(nrm2)
            for j in 1:ni
                W[i, j] += c * _dora_entry(W0, θ, i, j, oA, oB, no, r, s) - Float64(W0[i, j])
            end
        end
    end
    return out
end

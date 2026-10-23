"""Which components of [`TiedLoRA`](@ref) are trainable."""
Base.@kwdef struct TiedLoRAComponents
    A::Bool
    B::Bool
    u::Bool
    v::Bool
end

"""
    TiedLoRA(weights; rank, alpha, scaling, trainable, init, rng)

`ΔW = s·Λ_v·B·Λ_u·A` (Tied-LoRA, Renduchintala et al. 2024, arXiv:2311.09578, Eq. 1) with `A ∈ R^{r×n_in}` and
`B ∈ R^{n_out×r}` shared by every matrix and per-site `u ∈ R^r`, `v ∈ R^{n_out}`. `trainable` selects the trained
components (the paper's TL₂–TL₆ and LoRA/VeRA configurations); the rest stay at their initial values.
Initialization follows the paper's Table 1: `A` drawn from `init`, `u = 1`; if `v` is trained, `v = 0` and `B`
is drawn from `init`; otherwise, if `B` is trained, `B = 0` and `v = 1`. A configuration that trains neither `B`
nor `v` (TL₁) has `ΔW ≠ 0` at initialization and is rejected. All matrices must have the same shape.
`θ`: `vec(A)` if trained, `vec(B)` if trained, then per matrix `u` if trained and `v` if trained.
"""
struct TiedLoRA{
    FT <: AbstractFloat,
    VN <: AbstractVector{NTuple{2, Int}},
    VI <: AbstractVector{<:Integer},
    M <: AbstractMatrix{FT},
    V <: AbstractVector{FT},
} <: Adapter
    idx::VI
    shapes::VN
    site_offsets::VI
    n::Int
    rank::Int
    scale::FT
    trainable::TiedLoRAComponents
    A0::M
    B0::M
    u0::V
    v0::V
end

function TiedLoRA(
    base::AbstractVector{<:AbstractMatrix};
    rank::Integer,
    alpha::Real,
    scaling::LoRAScaling,
    trainable::TiedLoRAComponents,
    init::FactorInit,
    rng::Random.AbstractRNG,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1"))
    (trainable.B || trainable.v) ||
        throw(ArgumentError("Tied-LoRA with neither B nor v trained (TL1) has ΔW ≠ 0 at initialization"))
    idx, shapes = resolve_arrays(base)
    allequal(shapes) || throw(ArgumentError("Tied-LoRA sites must share one shape; got $(unique(shapes))"))
    no, ni = first(shapes)
    A0 = draw_factor(rng, init, rank, ni)
    B0 = trainable.v ? draw_factor(rng, init, no, rank) : zeros(no, rank)
    u0 = ones(rank)
    v0 = trainable.v ? zeros(no) : ones(no)
    shared = (trainable.A ? rank * ni : 0) + (trainable.B ? no * rank : 0)
    per_site = (trainable.u ? rank : 0) + (trainable.v ? no : 0)
    site_offsets = [shared + (k - 1) * per_site for k in eachindex(idx)]
    n = shared + per_site * length(idx)
    n >= 1 || throw(ArgumentError("Tied-LoRA with no trainable component"))
    return TiedLoRA(
        idx,
        shapes,
        site_offsets,
        n,
        Int(rank),
        float(lora_scale(scaling, alpha, rank)),
        trainable,
        A0,
        B0,
        u0,
        v0,
    )
end

nparams(a::TiedLoRA) = a.n

function segments(a::TiedLoRA)
    t, r = a.trainable, a.rank
    no, ni = first(a.shapes)
    out = Pair{Symbol, UnitRange{Int}}[]
    o = 0
    t.A && (push!(out, :A => (o + 1):(o + r * ni)); o += r * ni)
    t.B && (push!(out, :B => (o + 1):(o + no * r)); o += no * r)
    for _ in a.idx
        t.u && (push!(out, :u => (o + 1):(o + r)); o += r)
        t.v && (push!(out, :v => (o + 1):(o + no)); o += no)
    end
    return out
end

function reference_point(a::TiedLoRA)
    t = a.trainable
    parts = Vector{Float64}[]
    t.A && push!(parts, vec(a.A0))
    t.B && push!(parts, vec(a.B0))
    for _ in a.idx
        t.u && push!(parts, a.u0)
        t.v && push!(parts, a.v0)
    end
    return reduce(vcat, parts)
end

function add_delta!(out::AbstractVector{<:AbstractMatrix}, a::TiedLoRA, base::AbstractVector{<:AbstractMatrix}, θ::AbstractVector)
    check_inputs(a, out, base, θ)
    t, r, s = a.trainable, a.rank, a.scale
    no, ni = first(a.shapes)
    oB = t.A ? r * ni : 0
    @inbounds for (k, ix) in enumerate(a.idx)
        W = out[ix]
        ou = a.site_offsets[k]
        ov = ou + (t.u ? r : 0)
        for j in 1:ni, i in 1:no
            acc = zero(eltype(θ))
            for q in 1:r
                Aqj = t.A ? θ[q + (j - 1) * r] : a.A0[q, j]
                Biq = t.B ? θ[oB + i + (q - 1) * no] : a.B0[i, q]
                uq = t.u ? θ[ou + q] : a.u0[q]
                acc += Biq * uq * Aqj
            end
            vi = t.v ? θ[ov + i] : a.v0[i]
            W[i, j] += s * vi * acc
        end
    end
    return out
end

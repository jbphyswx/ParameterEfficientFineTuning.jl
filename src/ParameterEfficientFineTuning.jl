module ParameterEfficientFineTuning

using LowRankAdaptation: LowRankAdaptation, LowRankAdaptation as LoRA
using Random: Random

public LowRankAdaptation, LoRA
public Adapter, nparams, reference_point, segments, add_delta!, apply!
public Full, BitFit, RandomSubspace, projection, FourierFT, locations

include("Adapters.jl")

# Keep the standalone low-rank package independent while exposing the umbrella interface.
nparams(a::LoRA.Adapter) = LoRA.nparams(a)
reference_point(a::LoRA.Adapter) = LoRA.reference_point(a)
segments(a::LoRA.Adapter) = LoRA.segments(a)
check_arrays(a::LoRA.Adapter, arrays::AbstractVector{<:AbstractArray}) = LoRA.check_arrays(a, arrays)

apply!(out::AbstractVector{<:AbstractArray}, a::LoRA.Adapter, base::AbstractVector{<:AbstractArray}, θ::AbstractVector) =
    LoRA.apply!(out, a, base, θ)
add_delta!(out::AbstractVector{<:AbstractArray}, a::LoRA.Adapter, base::AbstractVector{<:AbstractArray}, θ::AbstractVector) =
    LoRA.add_delta!(out, a, base, θ)

include("Full.jl")
include("RandomSubspace.jl")
include("FourierFT.jl")

end

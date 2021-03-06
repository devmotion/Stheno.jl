export Circularised, lb, ub

"""
    Circularised{K<:Kernel} <: Kernel

A kernel `k` which only accepts data on the domain `lb(k)` to `ub(k)`. This condition is not
checked at run-time though, as it would imply an unacceptably large overhead. Note that this
is a hack that doesn't work in all situations as a covariance matrix generated by it is not
necessarily positive definite. You have been warned...
"""
struct Circularised{Tk<:Kernel, Tx<:AbstractFloat} <: Kernel
    k::Tk
    lb::Tx
    ub::Tx
    d::Tx
    twod::Tx
    Circularised(k::Tk, lb::Tx, ub::Tx) where {Tk<:Kernel, Tx<:AbstractFloat} =
        new{Tk, Tx}(k, lb, ub, (ub - lb) / 2, ub - lb)
end

lb(k::Circularised) = k.lb
ub(k::Circularised) = k.ub
(k::Circularised)(x::Real, y::Real) = k.k(mod(x - y + k.d, k.twod) - k.d, 0.0) # Definitely doing slightly more work than necessary here.
==(a::Circularised, b::Circularised) = (lb(a) == lb(b)) && (ub(a) == ub(b)) && (a.k == b.k)

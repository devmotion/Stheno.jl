using ToeplitzMatrices
import LinearAlgebra: Symmetric, *, mul!

import Base: transpose, adjoint, copy
import ToeplitzMatrices: Toeplitz

function copy(T::Toeplitz)
    return Toeplitz(copy(T.vc), copy(T.vr), copy(T.vcvr_dft), copy(T.tmp), T.dft)
end
function copy(T::SymmetricToeplitz)
    return SymmetricToeplitz(copy(T.vc), copy(T.vcvr_dft), copy(T.tmp), T.dft)
end

function +(T::SymmetricToeplitz, u::UniformScaling)
    Tu = copy(T)
    Tu.vc[1] += u.λ
    Tu.vcvr_dft .+= u.λ
    return Tu
end
+(u::UniformScaling, T::SymmetricToeplitz) = T + u

transpose(T::Toeplitz) = Toeplitz(T.vr, T.vc)
adjoint(T::Toeplitz) = Toeplitz(conj.(T.vr), conj.(T.vc))
transpose(T::SymmetricToeplitz) = T
adjoint(T::SymmetricToeplitz) = T

@inline LinearAlgebra.Symmetric(T::SymmetricToeplitz) = T

Toeplitz(vc::AbstractVector, vr::AbstractVector) = Toeplitz(Vector(vc), Vector(vr))

# """
#     mul!(C::Matrix, A::AbstractToeplitz, B::AbstractToeplitz)

# `O(prod(size(C)))` matrix multiplication for Toeplitz matrices. Follows from a skeleton of
# an algorithm on stackoverflow:
# https://stackoverflow.com/questions/15889521/product-of-two-toeplitz-matrices
# """
# function mul!(C::Matrix, A::AbstractToeplitz, B::AbstractToeplitz)
#     for q in 1:size(C, 2)
#         for p in 1:size(C, 1)
#             C[p, q]
#         end
#     end
#     return C
# end

# # function *(A::AbstractToeplitz, B::AbstractToeplitz)

# # end

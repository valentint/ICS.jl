##  using LinearAlgebra

"""
    sort_eigenvalues_eigenvectors(eigenvalues, eigenvectors)

Sort eigenvalues and eigenvectors in descending order of eigenvalues.

# Arguments
- `eigenvalues::AbstractVector`: Vector of eigenvalues.
- `eigenvectors::AbstractMatrix`: Corresponding eigenvectors.

# Returns
Tuple `(eigenvalues, eigenvectors)` sorted in descending order.
"""
function sort_eigenvalues_eigenvectors(eigenvalues::AbstractVector,
                                       eigenvectors::AbstractMatrix)
    idx = sortperm(eigenvalues; rev=true)
    return eigenvalues[idx], eigenvectors[:, idx]
end


"""
    sqrt_symmetric_matrix(A; inverse=false)

Compute the square root or inverse square root of a symmetric matrix.

# Arguments
- `A::AbstractMatrix`: Symmetric matrix.
- `inverse::Bool=false`: If true, compute the inverse square root.

# Returns
Matrix representing the (inverse) square root of `A`.
"""
function sqrt_symmetric_matrix(A::AbstractMatrix; inverse::Bool=false)
    F = eigen(Symmetric(A))   # eigen decomposition
    vals, vecs = sort_eigenvalues_eigenvectors(F.values, F.vectors)
    power = inverse ? -0.5 : 0.5
    return vecs * Diagonal(vals .^ power) * vecs'
end


"""
    check_gen_kurtosis(gen_kurtosis)

Check the `gen_kurtosis` array for NA, infinite, and complex values.
"""
function check_gen_kurtosis(gen_kurtosis::AbstractArray)
    if any(!isfinite, gen_kurtosis)
        @warn "Some generalized kurtosis values are infinite"
    end
    if any(!isreal, gen_kurtosis)
        @warn "Some generalized kurtosis values are complex"
    end
    if any(isnan, gen_kurtosis)
        @warn "Some generalized kurtosis values are NA (Not Available)"
    end
end


"""
    sign_max(row)

Determine the sign of the maximum absolute value in a row.

Returns:
- `1` if the maximum value equals the maximum absolute value (positive),
- `-1` otherwise.
"""
function sign_max(row::AbstractVector)
    return maximum(row) == maximum(abs.(row)) ? 1 : -1
end
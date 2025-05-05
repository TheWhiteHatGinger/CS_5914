#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Sparse>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using Triplet = Eigen::Triplet<double>;
using VectorXd = Eigen::VectorXd;

// Constructs a 2D Poisson matrix in CSR format
std::tuple<std::vector<double>, std::vector<int>, std::vector<int>, int, int> poisson(int nx, int ny) {
    int N = nx * ny;
    std::vector<Triplet> triplets;

    // Build the matrix using triplets
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            int idx = i * nx + j;
            // Main diagonal: 4
            triplets.emplace_back(idx, idx, 4.0);

            // Left neighbor: -1
            if (j > 0) {
                triplets.emplace_back(idx, idx - 1, -1.0);
            }
            // Right neighbor: -1
            if (j < nx - 1) {
                triplets.emplace_back(idx, idx + 1, -1.0);
            }
            // Up neighbor: -1
            if (i > 0) {
                triplets.emplace_back(idx, idx - nx, -1.0);
            }
            // Down neighbor: -1
            if (i < ny - 1) {
                triplets.emplace_back(idx, idx + nx, -1.0);
            }
        }
    }

    // Construct sparse matrix
    SpMat A(N, N);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    // Extract CSR components
    std::vector<double> values(A.nonZeros());
    std::vector<int> colind(A.nonZeros());
    std::vector<int> rowptr(A.outerSize() + 1);

    for (int i = 0; i < A.nonZeros(); ++i) {
        values[i] = A.valuePtr()[i];
        colind[i] = A.innerIndexPtr()[i];
    }
    for (int i = 0; i <= A.outerSize(); ++i) {
        rowptr[i] = A.outerIndexPtr()[i];
    }

    return {values, colind, rowptr, N, N};
}

// Conjugate Gradient solver
std::vector<double> cg_solve(const std::vector<double>& values,
                             const std::vector<int>& colind,
                             const std::vector<int>& rowptr,
                             int rows, int cols,
                             const std::vector<double>& b,
                             double tol, int maxiter) {
    // Reconstruct Eigen sparse matrix
    SpMat A(rows, cols);
    std::vector<Triplet> triplets;
    for (int i = 0; i < rows; ++i) {
        for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            triplets.emplace_back(i, colind[j], values[j]);
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    // Setup CG solver
    VectorXd b_eigen = Eigen::Map<const VectorXd>(b.data(), b.size());
    VectorXd x = VectorXd::Zero(b.size());
    VectorXd r = b_eigen - A * x;
    VectorXd p = r;
    double rsold = r.squaredNorm();

    if (maxiter == 0) {
        maxiter = b.size();
    }

    // CG iteration
    for (int i = 0; i < maxiter; ++i) {
        VectorXd Ap = A * p;
        double alpha = rsold / (p.dot(Ap));
        x += alpha * p;
        r -= alpha * Ap;
        double rsnew = r.squaredNorm();

        if (std::sqrt(rsnew) < tol) {
            break;
        }

        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    }

    // Convert result to std::vector
    std::vector<double> x_vec(x.data(), x.data() + x.size());
    return x_vec;
}

PYBIND11_MODULE(amg_solver, m) {
    m.def("poisson", &poisson, "Construct a 2D Poisson matrix in CSR format",
          py::arg("nx"), py::arg("ny"));
    m.def("cg_solve", &cg_solve, "Solve Ax=b using Conjugate Gradient",
          py::arg("values"), py::arg("colind"), py::arg("rowptr"),
          py::arg("rows"), py::arg("cols"), py::arg("b"),
          py::arg("tol") = 1e-8, py::arg("maxiter") = 0);
}

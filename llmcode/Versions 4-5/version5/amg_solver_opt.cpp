#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>
#include <vector>
#include <stdexcept>
#include <omp.h>
#include <algorithm>

namespace py = pybind11;

using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using Triplet = Eigen::Triplet<double>;
using VectorXd = Eigen::VectorXd;

// Constructs a 2D Poisson matrix in CSR format with parallel triplet assembly
std::tuple<std::vector<double>, std::vector<int>, std::vector<int>, int, int> poisson(int nx, int ny) {
    int N = nx * ny;
    std::vector<std::vector<Triplet>> triplets_per_thread(omp_get_max_threads());

    // Parallel triplet construction
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        triplets_per_thread[tid].reserve(N * 5 / omp_get_num_threads()); // Estimate max non-zeros per thread

        #pragma omp for collapse(2)
        for (int i = 0; i < ny; ++i) {
            for (int j = 0; j < nx; ++j) {
                int idx = i * nx + j;
                // Main diagonal: 4
                triplets_per_thread[tid].emplace_back(idx, idx, 4.0);

                // Left neighbor: -1
                if (j > 0) {
                    triplets_per_thread[tid].emplace_back(idx, idx - 1, -1.0);
                }
                // Right neighbor: -1
                if (j < nx - 1) {
                    triplets_per_thread[tid].emplace_back(idx, idx + 1, -1.0);
                }
                // Up neighbor: -1
                if (i > 0) {
                    triplets_per_thread[tid].emplace_back(idx, idx - nx, -1.0);
                }
                // Down neighbor: -1
                if (i < ny - 1) {
                    triplets_per_thread[tid].emplace_back(idx, idx + nx, -1.0);
                }
            }
        }
    }

    // Merge triplets
    std::vector<Triplet> triplets;
    size_t total_size = 0;
    for (const auto& t : triplets_per_thread) {
        total_size += t.size();
    }
    triplets.reserve(total_size);
    for (auto& t : triplets_per_thread) {
        triplets.insert(triplets.end(), t.begin(), t.end());
    }

    // Construct sparse matrix
    SpMat A(N, N);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    // Extract CSR components
    std::vector<double> values(A.nonZeros());
    std::vector<int> colind(A.nonZeros());
    std::vector<int> rowptr(A.outerSize() + 1);

    #pragma omp parallel for
    for (int i = 0; i < A.nonZeros(); ++i) {
        values[i] = A.valuePtr()[i];
        colind[i] = A.innerIndexPtr()[i];
    }
    #pragma omp parallel for
    for (int i = 0; i <= A.outerSize(); ++i) {
        rowptr[i] = A.outerIndexPtr()[i];
    }

    return {values, colind, rowptr, N, N};
}

// Simple smoothed aggregation coarsening (uniform grid coarsening by factor 2)
SpMat create_prolongation(int nx, int ny) {
    int nx_coarse = nx / 2;
    int ny_coarse = ny / 2;
    int N_fine = nx * ny;
    int N_coarse = nx_coarse * ny_coarse;
    std::vector<Triplet> triplets;

    for (int i = 0; i < ny_coarse; ++i) {
        for (int j = 0; j < nx_coarse; ++j) {
            int coarse_idx = i * nx_coarse + j;
            int fine_i = 2 * i;
            int fine_j = 2 * j;
            // Map coarse point to fine grid (simple injection)
            int fine_idx = fine_i * nx + fine_j;
            if (fine_idx < N_fine) {
                triplets.emplace_back(fine_idx, coarse_idx, 1.0);
            }
        }
    }

    SpMat P(N_fine, N_coarse);
    P.setFromTriplets(triplets.begin(), triplets.end());
    P.makeCompressed();
    return P;
}

// AMG V-cycle for smoothed aggregation
std::vector<double> amg_vcycle(const std::vector<double>& values,
                               const std::vector<int>& colind,
                               const std::vector<int>& rowptr,
                               int rows, int cols,
                               const std::vector<double>& b,
                               double tol, int maxiter, int nx, int ny) {
    // Reconstruct fine-level matrix
    SpMat A(rows, cols);
    std::vector<Triplet> triplets;
    for (int i = 0; i < rows; ++i) {
        for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            triplets.emplace_back(i, colind[j], values[j]);
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    VectorXd b_eigen = Eigen::Map<const VectorXd>(b.data(), b.size());
    VectorXd x = VectorXd::Zero(b.size());

    // V-cycle parameters
    int max_levels = std::min(4, static_cast<int>(std::log2(std::min(nx, ny)) - 2));
    std::vector<SpMat> A_levels = {A};
    std::vector<SpMat> P_levels;
    int current_nx = nx, current_ny = ny;

    // Build multigrid hierarchy
    for (int level = 0; level < max_levels - 1; ++level) {
        if (current_nx < 4 || current_ny < 4) break;
        SpMat P = create_prolongation(current_nx, current_ny);
        P_levels.push_back(P);
        SpMat R = P.transpose(); // Restriction is transpose of prolongation
        SpMat A_coarse = R * A_levels.back() * P;
        A_levels.push_back(A_coarse);
        current_nx /= 2;
        current_ny /= 2;
    }

    // V-cycle solver
    for (int iter = 0; iter < std::max(1, maxiter); ++iter) {
        VectorXd r = b_eigen - A * x;
        if (r.norm() < tol) break;

        // V-cycle
        std::vector<VectorXd> residuals = {r};
        std::vector<VectorXd> corrections(A_levels.size(), VectorXd::Zero(A_levels[0].rows()));

        // Downward leg (pre-smoothing and restriction)
        for (size_t level = 0; level < A_levels.size() - 1; ++level) {
            // Gauss-Seidel smoothing (2 iterations)
            VectorXd z = corrections[level];
            for (int gs = 0; gs < 2; ++gs) {
                for (int i = 0; i < A_levels[level].rows(); ++i) {
                    double sum = 0.0;
                    double diag = 0.0;
                    for (SpMat::InnerIterator it(A_levels[level], i); it; ++it) {
                        if (it.col() == i) {
                            diag = it.value();
                        } else {
                            sum += it.value() * z[it.col()];
                        }
                    }
                    if (diag != 0.0) {
                        z[i] = (residuals.back()[i] - sum) / diag;
                    }
                }
            }
            corrections[level] = z;
            residuals.push_back(P_levels[level].transpose() * (residuals.back() - A_levels[level] * z));
        }

        // Coarse level solve (direct or CG)
        VectorXd coarse_corr;
        if (A_levels.back().rows() < 100) {
            Eigen::SparseLU<SpMat> solver;
            solver.compute(A_levels.back());
            if (solver.info() != Eigen::Success) {
                throw std::runtime_error("SparseLU decomposition failed");
            }
            coarse_corr = solver.solve(residuals.back());
        } else {
            Eigen::ConjugateGradient<SpMat> cg;
            cg.compute(A_levels.back());
            cg.setTolerance(tol);
            cg.setMaxIterations(50);
            coarse_corr = cg.solve(residuals.back());
        }
        corrections.back() = coarse_corr;

        // Upward leg (prolongation and post-smoothing)
        for (int level = A_levels.size() - 2; level >= 0; --level) {
            corrections[level] += P_levels[level] * corrections[level + 1];
            VectorXd z = corrections[level];
            for (int gs = 0; gs < 2; ++gs) {
                for (int i = 0; i < A_levels[level].rows(); ++i) {
                    double sum = 0.0;
                    double diag = 0.0;
                    for (SpMat::InnerIterator it(A_levels[level], i); it; ++it) {
                        if (it.col() == i) {
                            diag = it.value();
                        } else {
                            sum += it.value() * z[it.col()];
                        }
                    }
                    if (diag != 0.0) {
                        z[i] = (residuals[level][i] - sum) / diag;
                    }
                }
            }
            corrections[level] = z;
        }

        x += corrections[0];
    }

    std::vector<double> x_vec(x.data(), x.data() + x.size());
    return x_vec;
}

// Optimized Conjugate Gradient solver (used for Ruge-Stuben or standalone)
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

    // Use Eigen's CG solver with parallelization
    Eigen::ConjugateGradient<SpMat> cg;
    cg.compute(A);
    cg.setTolerance(tol);
    cg.setMaxIterations(maxiter == 0 ? rows : maxiter);

    VectorXd b_eigen = Eigen::Map<const VectorXd>(b.data(), b.size());
    VectorXd x = cg.solve(b_eigen);

    std::vector<double> x_vec(x.data(), x.data() + x.size());
    return x_vec;
}

PYBIND11_MODULE(amg_solver_opt, m) {
    m.def("poisson", &poisson, "Construct a 2D Poisson matrix in CSR format",
          py::arg("nx"), py::arg("ny"));
    m.def("cg_solve", &cg_solve, "Solve Ax=b using Conjugate Gradient",
          py::arg("values"), py::arg("colind"), py::arg("rowptr"),
          py::arg("rows"), py::arg("cols"), py::arg("b"),
          py::arg("tol") = 1e-8, py::arg("maxiter") = 0);
    m.def("amg_vcycle", &amg_vcycle, "Solve Ax=b using AMG V-cycle",
          py::arg("values"), py::arg("colind"), py::arg("rowptr"),
          py::arg("rows"), py::arg("cols"), py::arg("b"),
          py::arg("tol") = 1e-8, py::arg("maxiter") = 10,
          py::arg("nx"), py::arg("ny"));
}

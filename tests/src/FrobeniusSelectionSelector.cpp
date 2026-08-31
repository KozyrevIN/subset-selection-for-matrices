#include <memory>
#include <random>
#include <vector>

#include <doctest/doctest.h>

#include <Eigen/Dense>

#include <MatSubset/FrobeniusSelectionSelector.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

namespace {

//! Exposes the protected starting set so its recurrences can be checked
//! against ground truth.
template <typename Scalar>
struct StartingSetProbe : MatSubset::FrobeniusPivotingBase<Scalar> {
    std::string getAlgorithmName() const override { return "probe"; }
    std::vector<Eigen::Index> selectSubsetImpl(const Eigen::MatrixX<Scalar> &,
                                               Eigen::Index,
                                               Eigen::Index *) override {
        return {};
    }
    using MatSubset::FrobeniusPivotingBase<Scalar>::selectStartingSet;
};

template <typename Scalar>
Eigen::MatrixX<Scalar> gaussian(Eigen::Index m, Eigen::Index n,
                                std::mt19937 &gen) {
    std::normal_distribution<double> dist(0, 1);
    Eigen::MatrixX<Scalar> X(m, n);
    for (Eigen::Index i = 0; i < m; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
            X(i, j) = static_cast<Scalar>(dist(gen));
        }
    }
    return X;
}

} // namespace

TEST_CASE_TEMPLATE("frobenius selection", Scalar, float, double) {

    std::unique_ptr<MatSubset::FrobeniusSelectionSelector<Scalar>> selector =
        std::make_unique<MatSubset::FrobeniusSelectionSelector<Scalar>>();

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "frobenius selection");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 5);

    // Check bounds
    check_bounds(selector.get(), 3, 5);

    // The starting set never forms (V_S V_S^T)^{-1} or rescans the scores: it
    // extends M by one row and column per pivot and advances d in place. Pin
    // both against what they are defined to be, since a wrong recurrence
    // still returns a plausible-looking index set.
    SUBCASE("starting set carries the exact inverse Gram and scores") {
        const double tol = std::is_same_v<Scalar, float> ? 1e-3 : 1e-10;
        std::mt19937 gen(2024);
        for (Eigen::Index m : {4, 9, 17}) {
            for (Eigen::Index n : {40, 157}) {
                Eigen::MatrixXd X = gaussian<double>(m, n, gen);
                Eigen::HouseholderQR<Eigen::MatrixXd> qr(X.transpose());
                Eigen::MatrixXd V0 =
                    (qr.householderQ() * Eigen::MatrixXd::Identity(n, m))
                        .transpose();

                Eigen::MatrixX<Scalar> V = V0.cast<Scalar>();
                Eigen::MatrixX<Scalar> M;
                Eigen::ArrayX<Scalar> d;
                StartingSetProbe<Scalar> probe;
                std::vector<Eigen::Index> indices =
                    probe.selectStartingSet(V, &M, &d);

                // M is the inverse Gram of the selected block, in the frame
                // the reflections leave it in.
                Eigen::MatrixXd V_S = V.leftCols(m).template cast<double>();
                Eigen::MatrixXd M_true = (V_S * V_S.transpose()).inverse();
                CHECK((M.template cast<double>() - M_true).norm() <=
                      tol * M_true.norm());

                // d_j = 1 + v_j^T M v_j is invariant under that rotation, so
                // check it on the original columns in the permuted order.
                Eigen::MatrixXd V_S0(m, m);
                for (Eigen::Index i = 0; i < m; ++i) {
                    V_S0.col(i) = V0.col(indices[static_cast<size_t>(i)]);
                }
                Eigen::MatrixXd M_orig = (V_S0 * V_S0.transpose()).inverse();
                for (Eigen::Index j = 0; j < n; ++j) {
                    const Eigen::VectorXd v =
                        V0.col(indices[static_cast<size_t>(j)]);
                    const double expected = 1.0 + v.dot(M_orig * v);
                    CHECK(std::abs(static_cast<double>(d(j)) - expected) <=
                          tol * expected);
                }

                // tr(M) is the objective the greedy goes on to minimise.
                CHECK(std::abs(static_cast<double>(M.trace()) -
                               M_orig.trace()) <= tol * M_orig.trace());
            }
        }
    }

    SUBCASE("selection is deterministic") {
        std::mt19937 gen(7);
        Eigen::MatrixX<Scalar> X = gaussian<Scalar>(11, 90, gen);
        for (Eigen::Index k : {11, 30, 89}) {
            CHECK(selector->selectSubset(X, k) == selector->selectSubset(X, k));
        }
    }
}

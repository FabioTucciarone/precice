#pragma once

#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/irange.hpp>
#include <fenv.h>
#include "mapping/RadialBasisFctBaseMapping.hpp"
#include <numeric>
#include "io/ExportVTU.hpp"
#include "mapping/RadialBasisFctSolver.hpp"
#include "mapping/config/MappingConfiguration.hpp"
#include "mapping/config/MappingConfigurationTypes.hpp"
#include "mesh/Mesh.hpp"
#include "precice/impl/Types.hpp"
#include "profiling/Event.hpp"
#include <iostream>
#include <fstream>

#define F_GREEDY 1
#define P_GREEDY 0

namespace precice {
namespace mapping {


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
class GreedyCholeskyMapping : public GreedyMapping<RADIAL_BASIS_FUNCTION_T> {

  using RadialBasisFctBaseMapping<RADIAL_BASIS_FUNCTION_T>::_basisFunction;
  using GreedyParameter = MappingConfiguration::GreedyParameter;
  using super = GreedyMapping<RADIAL_BASIS_FUNCTION_T>;

  using super::_log;
  using super::_greedyIDs;
  using super::_invCholeskyA;
  using super::_referenceResidualNorm;

public:

  GreedyCholeskyMapping(
    Mapping::Constraint     constraint,
    int                     dimensions,
    RADIAL_BASIS_FUNCTION_T function,
    std::array<bool, 3>     deadAxis,
    Polynomial              polynomial,
    GreedyParameter         greedyParameter);

  void computeMapping() final override;
  void mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) final override;
  void mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) final override;

  void clear() final override;

  std::string getName() const final override;

private:
  Eigen::MatrixXd _basisMatrix;
  Eigen::MatrixXd _choleskyA;

  void updateReorderRemove(const Eigen::MatrixXd &y);
  void reorderBasis(const Eigen::MatrixXd &y, const size_t removalN);

  virtual void buildInterpolationMatrices(const Eigen::MatrixXd &y, const Eigen::MatrixXd &startResidual, const size_t startIndex) override;

  void updateInverse(size_t startIndex);
  virtual Eigen::MatrixXd recalculateResidual(const Eigen::MatrixXd &y, size_t basisExtend) override;

  void solveConservative(const time::Sample &inData, Eigen::VectorXd &outData) const;
  void solveConsistent(const time::Sample &inData, Eigen::VectorXd &outData) const;
};


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
GreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T, BETA>::GreedyCholeskyMapping(
    Mapping::Constraint     constraint,
    int                     dimensions,
    RADIAL_BASIS_FUNCTION_T function,
    std::array<bool, 3>     deadAxis,
    Polynomial              polynomial,
    GreedyParameter         greedyParameter)
    : GreedyMapping<RADIAL_BASIS_FUNCTION_T>(constraint, dimensions, function, deadAxis, polynomial, greedyParameter)
{ }


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T, BETA>::computeMapping() {

  precice::profiling::Event e("map.greedy.computeMapping.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  super::computeMapping();
  _invCholeskyA = Eigen::MatrixXd();
  _basisMatrix.resize(super::_inSize, super::_basisSize);

  if constexpr (BETA != F_GREEDY) {
    buildInterpolationMatrices(Eigen::MatrixXd(), Eigen::MatrixXd(), 0);
  } else if (this->hasConstraint(Mapping::CONSERVATIVE)) {
    super::_nearestMapping.setMeshes(this->input(), this->output());
    super::_nearestMapping.computeMapping();
  }

  this->_hasComputedMapping = true;
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T, BETA>::buildInterpolationMatrices(const Eigen::MatrixXd &y, const Eigen::MatrixXd &startResidual, const size_t startIndex) {

  Eigen::VectorXd basisVector(super::_inSize);
  size_t initialSize = _greedyIDs.size();

  Eigen::MatrixXd residual = startResidual; // TODO: copy necessary?
  Eigen::VectorXd powerFunction;

  if constexpr (BETA != F_GREEDY) {
    _basisMatrix.resize(super::_inSize, super::_basisSize);
    powerFunction = Eigen::VectorXd(super::_inSize);
    powerFunction.fill(_basisFunction.evaluate(0));
  }
  // Erase indices that should be reselected: startIndex, ..., n-1
  _greedyIDs.erase(_greedyIDs.begin() + startIndex, _greedyIDs.end());
  
  // Iterative selection of new points
  for (size_t n = startIndex; n < super::_maxIter; ++n) {

    const auto [i, greedyValue] = this->template select<BETA>(residual, powerFunction);
    const auto x                = super::_inputMesh->vertices().at(i);

    super::updateKernelVector(x, boost::irange(0UL, super::_inSize), basisVector);
    basisVector -= _basisMatrix.block(0, 0, super::_inSize, n) * _basisMatrix.block(i, 0, 1, n).transpose();

    if (greedyValue < super::_tolerance || basisVector(i) <= 0 || n == super::_basisSize - 1) {
      if (greedyValue < super::_tolerance || basisVector(i) <= 0) {
        break;
      }
      super::calculateIncreasedNumberOfCenters();
      _basisMatrix.conservativeResize(super::_inSize, super::_basisSize);
    }
    _greedyIDs.push_back(i);

    const double invP = 1.0 / std::sqrt(basisVector(i));
    basisVector *= invP;
    _basisMatrix.col(n) = basisVector;

    if constexpr (BETA != F_GREEDY) {
      powerFunction -= (Eigen::VectorXd) basisVector.array().square();
      _basisMatrix.col(n) = basisVector;
    } else {
      const Eigen::RowVectorXd newtonCoefficient = residual.row(i) * invP;
      residual -= basisVector * newtonCoefficient;
    }

    PRECICE_DEBUG("Iteration: {}, greedyValue = {}\n", n + 1, greedyValue);
  }

  PRECICE_INFO("Finished greedy search. Reordering cholesky matrix.", _greedyIDs.size());

  _choleskyA = _basisMatrix(_greedyIDs, Eigen::seqN(0, _greedyIDs.size()));
  if constexpr (BETA == F_GREEDY) {
    _referenceResidualNorm = residual.squaredNorm();
    super::printFGreedyConclusion(initialSize, startIndex);
  }

  super::fillEvaluationMatrix(startIndex);
  if constexpr (BETA == F_GREEDY) {
    updateInverse(startIndex);
  }
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T, BETA>::reorderBasis(const Eigen::MatrixXd &y, const size_t removalN) { 

  size_t n = _greedyIDs.size();

  Eigen::VectorXd basisVector = Eigen::VectorXd::Zero(n);
  Eigen::MatrixXd residual = y(_greedyIDs, Eigen::all);
  Eigen::MatrixXd localBasisMatrix = Eigen::MatrixXd::Zero(n, n); 
  std::vector<int> reorderedIDs;
  reorderedIDs.reserve(n);

  for (size_t m = 0; m < n - removalN; ++m) {

    // i: local index in reordered _greedyIDs-space, j: global index in _inputMesh-space
    const auto [i, fMax] = super::select(residual);
    const size_t j       = _greedyIDs.at(i);
    const auto   x       = super::_inputMesh->vertices().at(j);

    super::updateKernelVector(x, _greedyIDs, basisVector);
    reorderedIDs.push_back(j);
    basisVector -= localBasisMatrix.block(0, 0, n, m) * localBasisMatrix.block(i, 0, 1, m).transpose();

    const double invP = 1.0 / std::sqrt(basisVector(i));
    basisVector *= invP;
    localBasisMatrix.col(m) = basisVector;

    const Eigen::RowVectorXd newtonCoefficient = residual.row(i) * invP;
    residual -= basisVector * newtonCoefficient;
  }

  size_t rebuildIndex = n;
  for (size_t i = 0; i < n; i++) {
    if (std::find(reorderedIDs.begin(), reorderedIDs.end(), _greedyIDs.at(i)) == reorderedIDs.end()) { 
      rebuildIndex = i;
      break;
    }
  }

  if (rebuildIndex != n) {
    buildInterpolationMatrices(y, recalculateResidual(y, rebuildIndex), rebuildIndex);
  }
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
Eigen::MatrixXd GreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T, BETA>::recalculateResidual(const Eigen::MatrixXd &y, size_t basisExtend) {

  PRECICE_ASSERT(basisExtend <= _greedyIDs.size());

  Eigen::MatrixXd residual = y;
  for (size_t i = 0; i < basisExtend; i++) {
    int j = _greedyIDs.at(i);
    const double invP = 1.0 / _basisMatrix(j, i);
    const Eigen::RowVectorXd newtonCoefficient = residual.row(j) * invP;
    residual -= _basisMatrix.col(i) * newtonCoefficient;
  }
  return residual;
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T, BETA>::updateReorderRemove(const Eigen::MatrixXd &y) {

  size_t n = _greedyIDs.size();

  if (n == 0) {
    buildInterpolationMatrices(y, y, 0);
  } else {
    int removalN = static_cast<int>(std::round(std::max(0.01 * n, 1.0)));
    reorderBasis(y, removalN);
  }
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T, BETA>::updateInverse(size_t n0) {

  const size_t n = _greedyIDs.size();
  if (_invCholeskyA.cols() < n) {
    size_t m = _invCholeskyA.cols();
    _invCholeskyA.conservativeResize(n, n);
    _invCholeskyA.block(0, m + 1, n, n - m - 1) = Eigen::MatrixXd::Zero(n, n - m - 1);
  }
  for (size_t i = n0; i < n; ++i) {
    const double invP = 1.0 / _choleskyA(i, i);
    _invCholeskyA.block(i, 0, 1, i).noalias() = -_choleskyA.row(i).block(0, 0, 1, i) * _invCholeskyA.block(0, 0, i, i).template triangularView<Eigen::Lower>();
    _invCholeskyA(i, i)                       = 1;
    _invCholeskyA.block(i, 0, 1, i + 1) *= invP;
  }
  //_invCholeskyA = utils::invertLowerTriangularBlockwise(_choleskyA);
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T, BETA>::solveConsistent(const time::Sample &inData, Eigen::VectorXd &outData) const {

  const Eigen::VectorXd &linearisedVectors = inData.values;

  Eigen::MatrixXd y = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, super::_inSize).transpose();
  Eigen::MatrixXd polynomialCoeffs;

  if (super::_usesPolynomial) {
    polynomialCoeffs = super::_qrDecomposedQ.solve(y);
    y -= super::_polyMatrixQ * polynomialCoeffs;
  }

  Eigen::MatrixXd interpolationCoeffs = y(super::_greedyIDs, Eigen::all);
  _choleskyA.triangularView<Eigen::Lower>().solveInPlace(interpolationCoeffs);
  _choleskyA.transpose().triangularView<Eigen::Upper>().solveInPlace(interpolationCoeffs);

  for (int d = 0; d < inData.dataDims; d++) {
    outData(Eigen::seqN(d, super::_outSize, inData.dataDims)) =  super::_kernelEval.block(0, 0, _greedyIDs.size(), super::_outSize).transpose() * interpolationCoeffs.col(d);
  }
  if (super::_usesPolynomial) {
    for (int d = 0; d < inData.dataDims; d++) {
      outData(Eigen::seqN(d, super::_outSize, inData.dataDims)) += super::_polyMatrixU * polynomialCoeffs.col(d);
    }
  }
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T, BETA>::solveConservative(const time::Sample &inData, Eigen::VectorXd &outData) const {

  const Eigen::VectorXd &linearisedVectors = inData.values;

  const size_t          n = super::_greedyIDs.size();
  const Eigen::MatrixXd y = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, super::_outSize).transpose();

  Eigen::MatrixXd greedySolution = super::_kernelEval.block(0, 0, n, super::_outSize) * y;
  _choleskyA.block(0, 0, n, n).triangularView<Eigen::Lower>().solveInPlace(greedySolution);
  _choleskyA.block(0, 0, n, n).transpose().triangularView<Eigen::Upper>().solveInPlace(greedySolution);

  Eigen::MatrixXd prediction = Eigen::MatrixXd::Zero(super::_inSize, inData.dataDims);
  prediction(super::_greedyIDs, Eigen::all) = greedySolution;

  if (super::_usesPolynomial) {
    const Eigen::MatrixXd epsilon = super::_polyMatrixU.transpose() * y - super::_polyMatrixQ.transpose() * prediction;
    const Eigen::MatrixXd polynomialContribution = super::_qrDecomposedQ.transpose().solve(epsilon);
    prediction += polynomialContribution;
  }
  for (int d = 0; d < inData.dataDims; d++) {
    outData(Eigen::seqN(d, super::_inSize, inData.dataDims)) = prediction.col(d);
  }
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T, BETA>::mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) {

  precice::profiling::Event mapConsistentEvent("map.greedy.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  if constexpr (BETA == F_GREEDY) {
    super::solveConsistentFGreedy(inData, outData);
  } else {
    solveConsistent(inData, outData);
  }
  mapConsistentEvent.addData("basisSize", super::_greedyIDs.size());
  mapConsistentEvent.addData("inSize", super::_inSize);
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T, BETA>::mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) {

  precice::profiling::Event mapConservativeEvent("map.greedy.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  if constexpr (BETA == F_GREEDY) {
    PRECICE_WARN("Conservative f-greedy is not recommended: Nearest Neighbor mapping is required to approximate the data for center selection.");
    super::_nearestMapping.mapConsistent(inData, outData);
    super::updateInterpolationMatrices(Eigen::Map<const Eigen::MatrixXd>(outData.data(), inData.dataDims, super::_inSize).transpose());
  }
  solveConservative(inData, outData);

  mapConservativeEvent.addData("basisSize", super::_greedyIDs.size());
  mapConservativeEvent.addData("inSize", super::_inSize);
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
std::string GreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T, BETA>::getName() const {
  return "global-greedy RBF (f-cholesky-cpu-executor)";
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T, BETA>::clear() {
  super::clear();
  _choleskyA   = Eigen::MatrixXd();
  _basisMatrix = Eigen::MatrixXd();
}

} // namespace mapping
} // namespace precice

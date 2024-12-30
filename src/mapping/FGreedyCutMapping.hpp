#pragma once
#include <Eigen/Dense>
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

namespace precice {
namespace mapping {

template <typename RADIAL_BASIS_FUNCTION_T>
class FGreedyCutMapping : public GreedyMapping<RADIAL_BASIS_FUNCTION_T> {

  using RadialBasisFctBaseMapping<RADIAL_BASIS_FUNCTION_T>::_basisFunction;
  using GreedyParameter = MappingConfiguration::GreedyParameter;
  using super = GreedyMapping<RADIAL_BASIS_FUNCTION_T>;

  using super::_log;
  using super::_greedyIDs;

public:

  FGreedyCutMapping(
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
  Eigen::MatrixXd _interpolationCoeffs;
  Eigen::MatrixXd _kernelMatrix;
  Eigen::MatrixXd _invCholeskyA;

  void updateResidualAndCoeffs(const Eigen::MatrixXd &y, Eigen::MatrixXd &interpolationCoeffs, Eigen::MatrixXd &residual, const size_t n0);
  void addKernelMatrixColumn(const size_t greedyIndex);

  void buildInterpolationMatrices(const Eigen::MatrixXd &y, const Eigen::MatrixXd &r0, const size_t n0);
  Eigen::MatrixXd recalculateResidualAndCoeffs(const Eigen::MatrixXd &y, const size_t basisExtend);

  void exchange(const Eigen::MatrixXd &y, size_t removalN);
  void updateInterpolationMatrices(const Eigen::MatrixXd &y);
};


template <typename RADIAL_BASIS_FUNCTION_T>
FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::FGreedyCutMapping(
    Mapping::Constraint     constraint,
    int                     dimensions,
    RADIAL_BASIS_FUNCTION_T function,
    std::array<bool, 3>     deadAxis,
    Polynomial              polynomial,
    GreedyParameter         greedyParameter)
    : GreedyMapping<RADIAL_BASIS_FUNCTION_T>(constraint, dimensions, function, deadAxis, polynomial, greedyParameter)
{ }


template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::addKernelMatrixColumn(const size_t greedyIndex) {

  const mesh::Mesh::VertexContainer &inputVertices = super::_inputMesh->vertices();

  const auto &v = inputVertices.at(_greedyIDs.at(greedyIndex)).rawCoords(); // greedyIndex = n - 1 = _greedyIDs.size() - 1

  for (size_t i = 0; i < super::_inSize; i++) {
    const auto & u = inputVertices.at(i).rawCoords();
    const double d = computeSquaredDifference(u, v, super::_activeAxis);

    _kernelMatrix(i, greedyIndex) = _basisFunction.evaluate(std::sqrt(d));
  }
}


// basisExtend = rebuild Index: nr Centers to use = n0 ## ## with Coeffs teste für basisExtend == 0
template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::MatrixXd FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::recalculateResidualAndCoeffs(const Eigen::MatrixXd &y, const size_t basisExtend) {
  PRECICE_ASSERT(basisExtend <= _greedyIDs.size());

  const Eigen::MatrixXd Cy = _invCholeskyA.block(0, 0, basisExtend, basisExtend).triangularView<Eigen::Lower>() * y(_greedyIDs, Eigen::all).block(0, 0, basisExtend, y.cols());
  _interpolationCoeffs.block(0, 0, basisExtend, y.cols()) = _invCholeskyA.block(0, 0, basisExtend, basisExtend).transpose().triangularView<Eigen::Upper>() * Cy;
  _interpolationCoeffs.block(basisExtend, 0, _interpolationCoeffs.rows() - basisExtend, y.cols()) = Eigen::MatrixXd::Zero(_interpolationCoeffs.rows() - basisExtend, y.cols());
  return (y - (_kernelMatrix.block(0, 0, super::_inSize, basisExtend) * _interpolationCoeffs.block(0, 0, basisExtend, y.cols()))).cwiseAbs();
}


// n0 = nr Greedy Centers to use - 1 ## ## updateResidualAndCoeffs
template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::updateResidualAndCoeffs(const Eigen::MatrixXd &y, Eigen::MatrixXd &interpolationCoeffs, Eigen::MatrixXd &residual, const size_t n0) {

  const Eigen::MatrixXd cy = _invCholeskyA.block(n0, 0, 1, n0 + 1) * y.transpose()(Eigen::all, _greedyIDs).transpose();
  _interpolationCoeffs.block(0, 0, n0 + 1, y.cols()) += _invCholeskyA.block(n0, 0, 1, n0 + 1).transpose() * cy;
  residual = (y - (_kernelMatrix.block(0, 0, super::_inSize, n0 + 1) * _interpolationCoeffs.block(0, 0, n0 + 1, y.cols()))).cwiseAbs();
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::computeMapping() {

  precice::profiling::Event e("map.f-greedy-cut.computeMapping.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  super::computeMapping();
  _invCholeskyA = Eigen::MatrixXd::Zero(super::_basisSize, super::_basisSize);
  _kernelMatrix = Eigen::MatrixXd::Zero(super::_inSize, super::_basisSize);
  this->_hasComputedMapping = true;
}

template <typename RADIAL_BASIS_FUNCTION_T> // beginne erneut bei Index n0 (= behalte n0 Zentren)
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::buildInterpolationMatrices(const Eigen::MatrixXd &y, const Eigen::MatrixXd &r0, const size_t n0) {

  Eigen::MatrixXd residual               = r0;
  Eigen::VectorXd kernelVectorOldCenters = Eigen::VectorXd::Ones(super::_basisSize);
  Eigen::VectorXd basisVector            = Eigen::VectorXd::Ones(super::_basisSize); // max(super::_basisSize, initialN)

   size_t initialN = _greedyIDs.size();
  _greedyIDs.erase(_greedyIDs.begin() + n0, _greedyIDs.end()); // n0 = 0 => Recalc everything

  _interpolationCoeffs.conservativeResize(super::_basisSize, r0.cols()); //TODO hier ??
  _interpolationCoeffs.block(n0, 0, _interpolationCoeffs.rows() - n0, r0.cols()) = Eigen::MatrixXd::Zero(_interpolationCoeffs.rows() - n0, r0.cols());

  const double kernelDiagonal = _basisFunction.evaluate(0);

  // Iterative selection of new points
  for (size_t n = n0; n < super::_maxIter; ++n) {

    const auto [i, fMax] = super::select(residual);
    const auto x         = super::_inputMesh->vertices().at(i);

    super::updateKernelVector(x, _greedyIDs, kernelVectorOldCenters);
    basisVector.head(n)  = _invCholeskyA.block(0, 0, n, n).triangularView<Eigen::Lower>() * kernelVectorOldCenters.head(n);
    const double squareP = kernelDiagonal - basisVector.array().head(n).square().sum();
    const double invP    = 1.0 / std::sqrt(squareP);

    if (fMax < super::_tolerance || n == super::_basisSize - 1) {
      if (fMax < super::_tolerance) 
        break;
      super::calculateIncreasedNumberOfCenters();
      _kernelMatrix.conservativeResize(super::_inSize, super::_basisSize);
      _invCholeskyA.conservativeResize(super::_basisSize, super::_basisSize);
      _invCholeskyA.block(0, n + 1, super::_basisSize, super::_basisSize - n - 1) = Eigen::MatrixXd::Zero(super::_basisSize, super::_basisSize - n - 1);
      kernelVectorOldCenters.conservativeResize(super::_basisSize);
      basisVector.conservativeResize(super::_basisSize);
      _interpolationCoeffs.conservativeResize(super::_basisSize, r0.cols());
      _interpolationCoeffs.block(n + 1, 0, super::_basisSize - n - 1, r0.cols()) = Eigen::MatrixXd::Zero(super::_basisSize - n - 1, r0.cols());
    }
    _greedyIDs.push_back(i);

    _invCholeskyA.block(n, 0, 1, n).noalias() = -basisVector.block(0, 0, n, 1).transpose() * _invCholeskyA.block(0, 0, n, n).triangularView<Eigen::Lower>();
    _invCholeskyA(n, n)                       = 1;
    _invCholeskyA.block(n, 0, 1, n + 1) *= invP;

    addKernelMatrixColumn(n);
    updateResidualAndCoeffs(y, _interpolationCoeffs, residual, n);

    PRECICE_DEBUG("Iteration: {}, fMax = {}\n", n + 1, fMax, squareP);
  }

  PRECICE_INFO("Finished greedy search and construction of inverse.");

  super::fillEvaluationMatrix(n0);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::updateInterpolationMatrices(const Eigen::MatrixXd &y) { //TODO: Code duplikat
  size_t n = _greedyIDs.size();

  if (n == 0) {
    buildInterpolationMatrices(y, y, 0);
  } else {
    int removalN = static_cast<int>(std::round(std::max(0.01 * n, 1.0))); // TODO: überdenken; min 1
    exchange(y, removalN);
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::exchange(const Eigen::MatrixXd &y, size_t removalN) { //TODO: Code duplikat

  size_t n = _greedyIDs.size();
  Eigen::MatrixXd _partialInverseA = Eigen::MatrixXd::Zero(2 * removalN - 1, 2 * removalN - 1);

  _interpolationCoeffs = _invCholeskyA.block(0, 0, n, n).triangularView<Eigen::Lower>() * y(_greedyIDs, Eigen::all);
  _interpolationCoeffs = _invCholeskyA.block(0, 0, n, n).transpose().triangularView<Eigen::Upper>() * _interpolationCoeffs;

  double minResidualNorm = std::numeric_limits<double>::max();
  double rebuildIndex = n;
  size_t blockHeight = removalN;

  for (size_t m = 0; m < n; m += blockHeight) {
    if (m + 2 * removalN > n) blockHeight = n - m;
    size_t blockLength = m + blockHeight;

    _partialInverseA.block(0, 0, blockHeight, blockHeight) = _invCholeskyA.block(m, 0, blockHeight, blockLength) * _invCholeskyA.block(m, 0, blockHeight, blockLength).transpose();
    double partialResidual = (_partialInverseA.block(0, 0, blockHeight, blockHeight).inverse() * _interpolationCoeffs.block(m, 0, blockHeight, _interpolationCoeffs.cols())).squaredNorm();

    if (partialResidual <= minResidualNorm) {
      minResidualNorm = partialResidual;
      rebuildIndex = m;
    }
  }

  if (rebuildIndex != n) {
    buildInterpolationMatrices(y, recalculateResidualAndCoeffs(y, rebuildIndex), rebuildIndex);
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) { //TODO: duplikat
  
  precice::profiling::Event mapConsistentEvent("map.f-greedy.computeMapping", profiling::Synchronize);
  precice::profiling::Event updateEvent("map.f-greedy.computeMapping.update", profiling::Synchronize);

  const Eigen::VectorXd &linearisedVectors = inData.values;

  Eigen::MatrixXd y = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, super::_inSize).transpose();

  Eigen::MatrixXd polynomialCoeffs;
  if (super::_usesPolynomial) {
    super::fillPolynomialMatrices();
    polynomialCoeffs = super::_qrDecomposedQ.solve(y);
    y -= super::_polyMatrixQ * polynomialCoeffs;
  }

  updateInterpolationMatrices(y);

  updateEvent.stop();

  precice::profiling::Event solveEvent("map.f-greedy.computeMapping.solve", profiling::Synchronize);

  size_t n = _greedyIDs.size();

  _interpolationCoeffs = _invCholeskyA.block(0,0,n,n).triangularView<Eigen::Lower>() * y(_greedyIDs, Eigen::all); //TODO: block for updateInverse(n0)
  _interpolationCoeffs = _invCholeskyA.block(0,0,n,n).transpose().triangularView<Eigen::Upper>() * _interpolationCoeffs;

  for (int d = 0; d < inData.dataDims; d++) {
    outData(Eigen::seqN(d, super::_outSize, inData.dataDims)) = super::_kernelEval.block(0, 0, n, super::_outSize).transpose() * _interpolationCoeffs.col(d);
  }
  if (super::_usesPolynomial) {
    for (int d = 0; d < inData.dataDims; d++) {
      outData(Eigen::seqN(d, super::_outSize, inData.dataDims)) += super::_polyMatrixU * polynomialCoeffs.col(d);
    }
  }

  solveEvent.stop();
  mapConsistentEvent.addData("basisSize", _greedyIDs.size());
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) {

  precice::profiling::Event e("map.f-greedy-cut.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  const Eigen::VectorXd &linearisedVectors = inData.values;
  Eigen::MatrixXd y = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, super::_inSize).transpose();
  updateInterpolationMatrices(y);
  super::solveConservativeWithCut(inData, _invCholeskyA, outData);
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::string FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::getName() const {
  return "global-greedy RBF (f-cut-cpu-executor)";
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::clear() {
  super::clear();
  _kernelMatrix = Eigen::MatrixXd();
  _invCholeskyA          = Eigen::MatrixXd();
}


} // namespace mapping
} // namespace precice

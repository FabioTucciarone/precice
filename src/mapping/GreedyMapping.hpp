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

namespace precice {
namespace mapping {

template <typename RADIAL_BASIS_FUNCTION_T>
class GreedyMapping : public RadialBasisFctBaseMapping<RADIAL_BASIS_FUNCTION_T> {

  using GreedyParameter = MappingConfiguration::GreedyParameter;

public:
  GreedyMapping(
    Mapping::Constraint     constraint,
    int                     dimensions,
    RADIAL_BASIS_FUNCTION_T function,
    std::array<bool, 3>     deadAxis,
    Polynomial              polynomial,
    GreedyParameter         greedyParameter);

  virtual void computeMapping() override;
  virtual void mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) = 0;
  virtual void mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) = 0;
  virtual void clear() override;

protected:
  using RadialBasisFctBaseMapping<RADIAL_BASIS_FUNCTION_T>::_basisFunction;

  precice::logging::Logger _log{"mapping::GreedyRBFMapping"};

  bool   _usesPolynomial;
  double _tolerance;
  size_t _maxIter;

  mesh::PtrMesh _inputMesh;
  mesh::PtrMesh _outputMesh;

  size_t _inSize    = 0;
  size_t _outSize   = 0;
  size_t _basisSize = 0;

  /// Indices on the input mesh selected by the greedy algorithm.
  std::vector<int>    _greedyIDs;
  std::array<bool, 3> _activeAxis;

  /// Inverse of the lower triangular matrix of the Cholesky-decomposition using the selected centers (_greedyIDs). 
  /// Used and updated by al f-greedy and cut methods after buildInterpolationMatrices() or computeMapping().
  Eigen::MatrixXd _invCholeskyA;

  // Matrices used by all greedy methods

  Eigen::MatrixXd _kernelEval;
  Eigen::MatrixXd _polyMatrixQ;
  Eigen::MatrixXd _polyMatrixU;

  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> _qrDecomposedQ;

  /// Residual of the last mapping. Used by the f-greedy methods and updated in buildInterpolationMatrices(). Might be used to decide whether to rebuild or not.
  double _referenceResidualNorm;

  enum UpdateMode {REBUILD, REBUILD_AT_TOLERANCE, EXCHANGE, EXCHANGE_AT_TOLERANCE}; // TODO: entfernen
  UpdateMode _updateMode = UpdateMode::EXCHANGE;

  /**
   * @brief select the next greedy-center.
   * @return Pair consisting of the input mesh index and the maximum value of the greedy criterion.
   */
  std::pair<int, double> select(const Eigen::VectorXd &powerFunction) const;

  /**
   * @brief select the next greedy-center.
   * 
   * This implementation calculates the square-norm of the rows of the input and finds the maximum.
   * 
   * @return Pair consisting of the input mesh index and the maximum value of the greedy criterion.
   */
  std::pair<int, double> select(const Eigen::MatrixXd &residual) const;

  /**
   * @brief Fills and potentially resizes the _kernelEval matrix used to evaluate the model on the output mesh keeping the entries for centers 0, ..., startIndex-1.
   * 
   * Builds _kernelEval of size _outSize x _greedyIDs.size() using the selected centers stored in _greedyIDs.
   * 
   * @param[in] startIndex Index with respect to _greedyIDs from wich to recompute the matrix entries.
   */
  void fillEvaluationMatrix(size_t startIndex);

  /**
   * @brief Build matrices _polyMatrixQ, _polyMatrixU and the QR-decomposition _qrDecomposedQ of _polyMatrixQ.
   */
  void fillPolynomialMatrices();

  /**
   * @brief Update
   * 
   * @param[in] x
   * @param[in] ids
   * @param[out] kernelVector
   */
  template <typename IndexContainer>
  void updateKernelVector(const mesh::Vertex &x, const IndexContainer &ids, Eigen::VectorXd &kernelVector) const;

  void solveConservativeWithCut(const time::Sample &inData, const Eigen::MatrixXd &cut, Eigen::VectorXd &outData) const;
  void solveConsistentWithCut(const time::Sample &inData, const Eigen::MatrixXd &cut, Eigen::VectorXd &outData) const;
  void solveConservativeWithCholesky(const time::Sample &inData, const Eigen::MatrixXd &choleskyA, Eigen::VectorXd &outData) const;
  void solveConsistentWithCholesky(const time::Sample &inData, const Eigen::MatrixXd &choleskyA, Eigen::VectorXd &outData) const;
  void solveConsistentFGreedy(const time::Sample &inData, Eigen::VectorXd &outData);

  size_t estimateNumberOfCenters();
  void calculateIncreasedNumberOfCenters();

  /**
   * @brief Updates the mapping for f-greedy methods by calling \ref buildInterpolationMatrices() or \ref exchange().
   */
  virtual void updateInterpolationMatrices(const Eigen::MatrixXd &y);

  /**
   * @brief Recomputes the mapping for f-greedy methods starting the center selection from startIndex.
   * 
   * Modifies and updates _inverseCholeskyA, _greedyIDs, _evalMatrix and possibly other datastructures.
   * Access solutions with Eigen::MatirxXd::block. _greedyIDs.size() is the number of selected centers.
   * 
   * @param[in] y The input samples as a matrix of the format (input size x data dimensions).
   * @param[in] startResidual The residual from which to start the center selection, meaning, the residual of the model up to the center index startIndex.
   * @param[in] startIndex Defines how many centers of the previouse model should be kept: 0, 1, ..., startIndex-1.
   */
  virtual void buildInterpolationMatrices(const Eigen::MatrixXd &y, const Eigen::MatrixXd &startResidual, const size_t startIndex);

  /**
   * @brief Recalculates the residual of the model using only the first m centers with m = basisExtent.
   */
  virtual Eigen::MatrixXd recalculateResidual(const Eigen::MatrixXd &y, const size_t basisExtent);

  /**
   * @brief Removes al least removalN centers deemed "less important" and adds back the rest by calling \ref buildInterpolationMatrices()
   * 
   * Splits the _greedyIDs list into mostly equaly sized divisions and removes the division with the lowest impact on the residual and all of the following ones.
   * Uses _inverseCholeskyA and the current y for the extended version of Rippa's algorithm to find the division to remove.
   * 
   * @param[in] removalN min. size of divisions of _greedyIDs for the removal with the extended Rippa's algorithm.
   * @param[in] y The input samples as a matrix of the format (input size x data dimensions).
   */
  void exchange(const Eigen::MatrixXd &y, size_t removalN);

  void printFGreedyConclusion(int initialN, int rebuildIndex);
};


template <typename RADIAL_BASIS_FUNCTION_T>
GreedyMapping<RADIAL_BASIS_FUNCTION_T>::GreedyMapping(
    Mapping::Constraint     constraint,
    int                     dimensions,
    RADIAL_BASIS_FUNCTION_T function,
    std::array<bool, 3>     deadAxis,
    Polynomial              polynomial,
    GreedyParameter         greedyParameter)
    : RadialBasisFctBaseMapping<RADIAL_BASIS_FUNCTION_T>(constraint, dimensions, function, deadAxis, Mapping::InitialGuessRequirement::None)
{
  PRECICE_CHECK(polynomial != Polynomial::ON, "Integrated polynomials not supported for greedy rbf methods");
  PRECICE_CHECK(greedyParameter.maxIterations > 0, "Maximum number of iterations cannot be smaller than 1.");
  _usesPolynomial = (polynomial == Polynomial::SEPARATE);

  _tolerance = greedyParameter.tolerance;
  _maxIter   = greedyParameter.maxIterations; // later: _maxIter := max(_inSize, _maxIter)

  if (greedyParameter.fUpdateMode == "exchange") _updateMode = UpdateMode::EXCHANGE;
  else if (greedyParameter.fUpdateMode == "tolerance-rebuild") _updateMode = UpdateMode::REBUILD_AT_TOLERANCE;
  else if (greedyParameter.fUpdateMode == "tolerance-exchange") _updateMode = UpdateMode::EXCHANGE_AT_TOLERANCE;
  else _updateMode = UpdateMode::REBUILD;
  PRECICE_INFO("f-greedy update mode is \"{}\"", greedyParameter.fUpdateMode);

  _activeAxis = std::array<bool, 3>({{false, false, false}});
  std::transform(deadAxis.begin(), deadAxis.end(), _activeAxis.begin(), [](const auto ax) { return !ax; });
}

template <typename RADIAL_BASIS_FUNCTION_T>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::fillPolynomialMatrices() {

  precice::profiling::Event e("fillPolynomialMatrices", profiling::Synchronize);

  unsigned int polyParams = 4 - std::count(_activeAxis.begin(), _activeAxis.end(), false);
  _polyMatrixQ.resize(_inSize, polyParams);
  fillPolynomialEntries(_polyMatrixQ, *_inputMesh, boost::irange((size_t) 0, _inSize), 0, _activeAxis);
  _polyMatrixU.resize(_outSize, polyParams);
  fillPolynomialEntries(_polyMatrixU, *_outputMesh, boost::irange((size_t) 0, _outSize), 0, _activeAxis);

  _qrDecomposedQ = _polyMatrixQ.colPivHouseholderQr();
}

template <typename RADIAL_BASIS_FUNCTION_T>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::fillEvaluationMatrix(size_t startIndex) {

  precice::profiling::Event e("fillEvaluationMatrix", profiling::Synchronize);

  const mesh::Mesh::VertexContainer &inputVertices  = _inputMesh->vertices();
  const mesh::Mesh::VertexContainer &outputVertices = _outputMesh->vertices();

  if (size_t(_kernelEval.rows()) < _greedyIDs.size() || size_t(_kernelEval.cols()) < _outSize) {
    _kernelEval.conservativeResize(_greedyIDs.size(), _outSize);
  }

  for (size_t i = startIndex; i < _greedyIDs.size(); i++) {
    const auto &u = inputVertices.at(_greedyIDs.at(i)).rawCoords();
    for (size_t j = 0; j < _outSize; j++) {
      const auto & v    = outputVertices.at(j).rawCoords();
      const double d    = computeSquaredDifference(u, v, _activeAxis);
      _kernelEval(i, j) = _basisFunction.evaluate(std::sqrt(d));
    }
  }
}

template <typename RADIAL_BASIS_FUNCTION_T> 
template<typename IndexContainer>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::updateKernelVector(const mesh::Vertex &x, const IndexContainer &ids, Eigen::VectorXd &kernelVector) const {

  const mesh::Mesh::VertexContainer &inputVertices = _inputMesh->vertices();
  for (const auto &j : ids | boost::adaptors::indexed()) {
    const auto &y   = inputVertices.at(j.value()).rawCoords();
    kernelVector(j.index()) = _basisFunction.evaluate(std::sqrt(computeSquaredDifference(x.rawCoords(), y, _activeAxis)));
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
size_t GreedyMapping<RADIAL_BASIS_FUNCTION_T>::estimateNumberOfCenters() {
  return static_cast<size_t>(0.1 * _inSize + 1);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::calculateIncreasedNumberOfCenters() {
  _basisSize = _basisSize + std::min(_inSize, static_cast<size_t>(0.1 * _inSize + 1));
  PRECICE_INFO("Resizing matrices to {}% ({}) of centers.", static_cast<size_t>((float(_basisSize) / float(_inSize)) * 100), _basisSize);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::computeMapping() {
  PRECICE_ASSERT(_greedyIDs.empty());
  PRECICE_ASSERT(_kernelEval.size() == 0);

  if (this->hasConstraint(Mapping::CONSERVATIVE)) {
    _inputMesh  = this->output();
    _outputMesh = this->input();
  } else {
    _inputMesh  = this->input();
    _outputMesh = this->output();
  }
  _inSize  = _inputMesh->vertices().size();
  _outSize = _outputMesh->vertices().size();

  _maxIter   = std::min(_inSize, _maxIter); // max iterations must be smaller than or equal to the number of verticies
  _basisSize = estimateNumberOfCenters();
  _greedyIDs.reserve(_basisSize);
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::pair<int, double> GreedyMapping<RADIAL_BASIS_FUNCTION_T>::select(const Eigen::VectorXd &powerFunction) const {
  Eigen::Index maxIndex;
  double       maxValue = powerFunction.maxCoeff(&maxIndex);
  return {maxIndex, maxValue};
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::pair<int, double> GreedyMapping<RADIAL_BASIS_FUNCTION_T>::select(const Eigen::MatrixXd &residual) const {
  Eigen::Index maxIndex;
  double       maxValue = residual.rowwise().squaredNorm().maxCoeff(&maxIndex);
  return {maxIndex, maxValue};
}

template <typename RADIAL_BASIS_FUNCTION_T>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::solveConservativeWithCut(const time::Sample &inData, const Eigen::MatrixXd &cut, Eigen::VectorXd &outData) const {
  const Eigen::VectorXd &linearisedVectors = inData.values;

  const size_t          n = _greedyIDs.size();
  const Eigen::MatrixXd y = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, _outSize).transpose();

  Eigen::MatrixXd u = _kernelEval.block(0, 0, n, _outSize) * y;
  Eigen::MatrixXd Cu = cut.block(0, 0, n, n).triangularView<Eigen::Lower>() * u;
  Eigen::MatrixXd greedySolution = (cut.block(0, 0, n, n).transpose().triangularView<Eigen::Upper>() * Cu)(_greedyIDs, Eigen::all);

  Eigen::MatrixXd prediction = Eigen::MatrixXd::Zero(_inSize, inData.dataDims);
  prediction(_greedyIDs, Eigen::all) = greedySolution;

  if (_usesPolynomial) {
    const Eigen::MatrixXd epsilon = _polyMatrixU.transpose() * y - _polyMatrixQ.transpose() * prediction;
    const Eigen::MatrixXd polynomialContribution = _qrDecomposedQ.transpose().solve(epsilon);
    prediction += polynomialContribution;
  }
  for (int d = 0; d < inData.dataDims; d++) {
    outData(Eigen::seqN(d, _inSize, inData.dataDims)) = prediction.col(d);
  }
}


template <typename RADIAL_BASIS_FUNCTION_T>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::exchange(const Eigen::MatrixXd &y, size_t removalN) {
  PRECICE_ASSERT(_invCholeskyA.cols() == _invCholeskyA.rows() && _invCholeskyA.rows() > 0);
  PRECICE_ASSERT(_greedyIDs.size() <= size_t(_invCholeskyA.cols()));
  PRECICE_ASSERT(_greedyIDs.size() >= removalN && removalN > 0);

  size_t n = _greedyIDs.size();
  Eigen::MatrixXd partialInverseA = Eigen::MatrixXd::Zero(2 * removalN - 1, 2 * removalN - 1);

  Eigen::MatrixXd interpolationCoeffs = _invCholeskyA.block(0, 0, n, n).triangularView<Eigen::Lower>() * y(_greedyIDs, Eigen::all);
  interpolationCoeffs = _invCholeskyA.block(0, 0, n, n).transpose().triangularView<Eigen::Upper>() * interpolationCoeffs;

  double minResidualNorm = std::numeric_limits<double>::max();
  double rebuildIndex = n;
  size_t blockHeight = removalN;

  for (size_t m = 0; m < n; m += blockHeight) {
    if (m + 2 * removalN > n) blockHeight = n - m;
    size_t blockLength = m + blockHeight;

    partialInverseA.block(0, 0, blockHeight, blockHeight) = _invCholeskyA.block(m, 0, blockHeight, blockLength) * _invCholeskyA.block(m, 0, blockHeight, blockLength).transpose();
    double partialResidual = (partialInverseA.block(0, 0, blockHeight, blockHeight).inverse() * interpolationCoeffs.block(m, 0, blockHeight, interpolationCoeffs.cols())).squaredNorm();

    if (partialResidual <= minResidualNorm) {
      minResidualNorm = partialResidual;
      rebuildIndex = m;
    }
  }

  if (rebuildIndex != n) {
    buildInterpolationMatrices(y, recalculateResidual(y, rebuildIndex), rebuildIndex);
  }
}


template <typename RADIAL_BASIS_FUNCTION_T>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::solveConservativeWithCholesky(const time::Sample &inData, const Eigen::MatrixXd &choleskyA, Eigen::VectorXd &outData) const {
  const Eigen::VectorXd &linearisedVectors = inData.values;

  const size_t          n = _greedyIDs.size();
  const Eigen::MatrixXd y = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, _outSize).transpose();

  Eigen::MatrixXd greedySolution = _kernelEval.block(0, 0, n, _outSize) * y;
  choleskyA.block(0, 0, n, n).triangularView<Eigen::Lower>().solveInPlace(greedySolution);
  choleskyA.block(0, 0, n, n).transpose().triangularView<Eigen::Upper>().solveInPlace(greedySolution);

  Eigen::MatrixXd prediction = Eigen::MatrixXd::Zero(_inSize, inData.dataDims);
  prediction(_greedyIDs, Eigen::all) = greedySolution;

  if (_usesPolynomial) {
    const Eigen::MatrixXd epsilon = _polyMatrixU.transpose() * y - _polyMatrixQ.transpose() * prediction;
    const Eigen::MatrixXd polynomialContribution = _qrDecomposedQ.transpose().solve(epsilon);
    prediction += polynomialContribution;
  }
  for (int d = 0; d < inData.dataDims; d++) {
    outData(Eigen::seqN(d, _inSize, inData.dataDims)) = prediction.col(d);
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::solveConsistentWithCut(const time::Sample &inData, const Eigen::MatrixXd &cut, Eigen::VectorXd &outData) const {
  const Eigen::VectorXd &linearisedVectors = inData.values;

  const size_t    n = _greedyIDs.size();
  Eigen::MatrixXd y = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, _inSize).transpose();
  Eigen::MatrixXd polynomialCoeffs;

  if (_usesPolynomial) {
    polynomialCoeffs = _qrDecomposedQ.solve(y);
    y -= _polyMatrixQ * polynomialCoeffs;
  }

  const Eigen::MatrixXd z = y(_greedyIDs, Eigen::all);
  const Eigen::MatrixXd Cz = cut.block(0, 0, n, n).triangularView<Eigen::Lower>() * z;
  const Eigen::MatrixXd interpolationCoeffs = cut.block(0, 0, n, n).transpose().triangularView<Eigen::Upper>() * Cz;

  for (int d = 0; d < inData.dataDims; d++) {
    outData(Eigen::seqN(d, _outSize, inData.dataDims)) = _kernelEval.block(0, 0, n, _outSize).transpose() * interpolationCoeffs.col(d);
  }
  if (_usesPolynomial) {
    for (int d = 0; d < inData.dataDims; d++) {
      outData(Eigen::seqN(d, _outSize, inData.dataDims)) += _polyMatrixU * polynomialCoeffs.col(d);
    }
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::solveConsistentWithCholesky(const time::Sample &inData, const Eigen::MatrixXd &choleskyA, Eigen::VectorXd &outData) const {
  const Eigen::VectorXd &linearisedVectors = inData.values;

  Eigen::MatrixXd y = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, _inSize).transpose();
  Eigen::MatrixXd polynomialCoeffs;

  if (_usesPolynomial) {
    polynomialCoeffs = _qrDecomposedQ.solve(y);
    y -= _polyMatrixQ * polynomialCoeffs;
  }

  Eigen::MatrixXd interpolationCoeffs = y(_greedyIDs, Eigen::all);
  choleskyA.triangularView<Eigen::Lower>().solveInPlace(interpolationCoeffs);
  choleskyA.transpose().triangularView<Eigen::Upper>().solveInPlace(interpolationCoeffs);

  for (int d = 0; d < inData.dataDims; d++) {
    outData(Eigen::seqN(d, _outSize, inData.dataDims)) = _kernelEval.block(0, 0, _greedyIDs.size(), _outSize).transpose() * interpolationCoeffs.col(d);
  }
  if (_usesPolynomial) {
    for (int d = 0; d < inData.dataDims; d++) {
      outData(Eigen::seqN(d, _outSize, inData.dataDims)) += _polyMatrixU * polynomialCoeffs.col(d);
    }
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::solveConsistentFGreedy(const time::Sample &inData, Eigen::VectorXd &outData) {
  PRECICE_ASSERT(_greedyIDs.size() <= size_t(_invCholeskyA.cols()));

  precice::profiling::Event mapConsistentEvent("map.f-greedy.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);
  precice::profiling::Event updateEvent("map.f-greedy.update", profiling::Synchronize);

  const Eigen::VectorXd &linearisedVectors = inData.values;
  Eigen::MatrixXd y = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, _inSize).transpose();

  Eigen::MatrixXd polynomialCoeffs;
  if (_usesPolynomial) {
    fillPolynomialMatrices();
    polynomialCoeffs = _qrDecomposedQ.solve(y);
    y -= _polyMatrixQ * polynomialCoeffs;
  }

  updateInterpolationMatrices(y);
  PRECICE_ASSERT(_invCholeskyA.cols() == _invCholeskyA.rows() && _invCholeskyA.cols() > 0);

  updateEvent.stop();

  precice::profiling::Event solveEvent("map.f-greedy.solve", profiling::Synchronize);

  size_t n = _greedyIDs.size();

  Eigen::MatrixXd interpolationCoeffs = _invCholeskyA.block(0,0,n,n).triangularView<Eigen::Lower>() * y(_greedyIDs, Eigen::all); //TODO: block for updateInverse(n0)
  interpolationCoeffs = _invCholeskyA.block(0,0,n,n).transpose().triangularView<Eigen::Upper>() * interpolationCoeffs;

  for (int d = 0; d < inData.dataDims; d++) {
    outData(Eigen::seqN(d, _outSize, inData.dataDims)) = _kernelEval.block(0, 0, n, _outSize).transpose() * interpolationCoeffs.col(d);
  }
  if (_usesPolynomial) {
    for (int d = 0; d < inData.dataDims; d++) {
      outData(Eigen::seqN(d, _outSize, inData.dataDims)) += _polyMatrixU * polynomialCoeffs.col(d);
    }
  }

  solveEvent.stop();
  mapConsistentEvent.addData("basisSize", _greedyIDs.size());
  mapConsistentEvent.addData("inSize", _inSize);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::buildInterpolationMatrices(const Eigen::MatrixXd &y, const Eigen::MatrixXd &r0, const size_t n0) { 
  PRECICE_ASSERT(false, "Not implemented!"); 
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::MatrixXd GreedyMapping<RADIAL_BASIS_FUNCTION_T>::recalculateResidual(const Eigen::MatrixXd &y, const size_t basisExtent) { 
  PRECICE_ASSERT(false, "Not implemented!"); 
}

template <typename RADIAL_BASIS_FUNCTION_T>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::updateInterpolationMatrices(const Eigen::MatrixXd &y) {
  size_t n = _greedyIDs.size();
  if (n == 0) {
    buildInterpolationMatrices(y, y, 0);
  } else {
    int removalN = static_cast<int>(std::round(std::max(0.01 * n, 1.0))); // TODO: überdenken; min 1

    switch (_updateMode) {
      case UpdateMode::EXCHANGE: {
        exchange(y, removalN);
        break;
      }
      case UpdateMode::REBUILD_AT_TOLERANCE: {
        double residualNorm = recalculateResidual(y, n).squaredNorm();
        if (residualNorm > 2 * _referenceResidualNorm) {
          buildInterpolationMatrices(y, y, 0);
        }
        break;
      }
      case UpdateMode::EXCHANGE_AT_TOLERANCE: {
        double residualNorm = recalculateResidual(y, n).squaredNorm();
        if (residualNorm > 2 * _referenceResidualNorm) {
          exchange(y, removalN);
        }
        break;
      }
      default: {
        buildInterpolationMatrices(y, y, 0);
        break;
      }
    }
  }
}


template <typename RADIAL_BASIS_FUNCTION_T>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::printFGreedyConclusion(int initialN, int rebuildIndex) { 
  
  int difference = int(_greedyIDs.size()) - initialN;
  double removalPercentage = std::round(float(initialN - rebuildIndex) / initialN * 10000) / 100.0;
  removalPercentage = (removalPercentage != removalPercentage) ? 0 : removalPercentage; // replace NaN with 0%

  PRECICE_INFO("Number of centers used now: {}, in previouse time step: {} (removed {}% in exchange-step, added {} in greedy-step)\n",
    _greedyIDs.size(), initialN, removalPercentage, difference
  );
}

template <typename RADIAL_BASIS_FUNCTION_T>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::clear() {
  _greedyIDs.clear();
  _kernelEval = Eigen::MatrixXd();
  _inSize     = 0;
  _outSize    = 0;
  _invCholeskyA = Eigen::MatrixXd();
}

} // namespace mapping
} // namespace precice

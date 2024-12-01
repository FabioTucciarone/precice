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

  bool _usesPolynomial;

  mesh::PtrMesh _inputMesh;
  mesh::PtrMesh _outputMesh;

  size_t _maxIter;
  double _tolerance;

  size_t _inSize  = 0;
  size_t _outSize = 0;
  size_t _basisSize = 0;

  std::vector<int>    _greedyIDs;
  std::array<bool, 3> _activeAxis;

  Eigen::MatrixXd _kernelEval;
  Eigen::MatrixXd _polyMatrixQ;
  Eigen::MatrixXd _polyMatrixU;

  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> _qrDecomposedQ;

  std::pair<int, double> select(const Eigen::VectorXd &powerFunction) const;

  void fillEvaluationMatrix();
  void fillPolynomialMatrices();

  template <typename IndexContainer>
  void updateKernelVector(const mesh::Vertex &x, const IndexContainer &ids, Eigen::VectorXd &kernelVector) const;

  void solveConservativeWithCut(const time::Sample &inData, const Eigen::MatrixXd &cut, Eigen::VectorXd &outData) const;
  void solveConsistentWithCut(const time::Sample &inData, const Eigen::MatrixXd &cut, Eigen::VectorXd &outData) const;
  void solveConservativeWithCholesky(const time::Sample &inData, const Eigen::MatrixXd &choleskyA, Eigen::VectorXd &outData) const;
  void solveConsistentWithCholesky(const time::Sample &inData, const Eigen::MatrixXd &choleskyA, Eigen::VectorXd &outData) const;

  size_t estimateNumberOfCenters();
  void calculateIncreasedNumberOfCenters();
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
  _maxIter   = greedyParameter.maxIterations;

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
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::fillEvaluationMatrix() {

  precice::profiling::Event e("fillEvaluationMatrix", profiling::Synchronize);

  const mesh::Mesh::VertexContainer &inputVertices  = _inputMesh->vertices();
  const mesh::Mesh::VertexContainer &outputVertices = _outputMesh->vertices();
  _kernelEval.resize(_greedyIDs.size(), outputVertices.size());

  for (size_t i = 0; i < _greedyIDs.size(); i++) {
    const auto &u = inputVertices.at(_greedyIDs.at(i)).rawCoords();
    for (size_t j = 0; j < outputVertices.size(); j++) {
      const auto & v    = outputVertices.at(j).rawCoords();
      const double d    = computeSquaredDifference(u, v, _activeAxis);
      _kernelEval(i, j) = _basisFunction.evaluate(std::sqrt(d));
    }
  }
}

template <typename RADIAL_BASIS_FUNCTION_T> 
template<typename IndexContainer>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::updateKernelVector(const mesh::Vertex &x, const IndexContainer &ids, Eigen::VectorXd &kernelVector) const {

  precice::profiling::Event e("updateKernelVector", profiling::Synchronize);

  const mesh::Mesh::VertexContainer &inputVertices = _inputMesh->vertices();
  for (const auto &j : ids | boost::adaptors::indexed()) {
    const auto &y   = inputVertices.at(j.value()).rawCoords();
    kernelVector(j.index()) = _basisFunction.evaluate(std::sqrt(computeSquaredDifference(x.rawCoords(), y, _activeAxis)));
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
size_t GreedyMapping<RADIAL_BASIS_FUNCTION_T>::estimateNumberOfCenters() {
  // auto x0 = _inputMesh->vertices().at(0);
  // std::vector<int> matches = _inputMesh->index().getClosestVertices(x0.getCoords(), 4);
  // double h = 0;
  // for (int i = 0; i < 3; i++) {
  //   auto xi = _inputMesh->vertices().at(matches.at(i));
  //   double h = std::sqrt(computeSquaredDifference(xi.rawCoords(), x0.rawCoords(), _activeAxis));
  //   h += std::sqrt(computeSquaredDifference(xi.rawCoords(), x0.rawCoords(), _activeAxis));
  // }
  // h /= 3;
  return static_cast<size_t>(0.1 * _maxIter + 1);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::calculateIncreasedNumberOfCenters() {
  _basisSize = _basisSize + std::min(_maxIter, static_cast<size_t>(0.1 * _maxIter + 1));
  PRECICE_INFO("Resizing matrices to {}% ({}) of centers.", static_cast<size_t>((float(_basisSize) / float(_inSize)) * 100), _basisSize);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::computeMapping() {
  PRECICE_ASSERT(_greedyIDs.empty());
  PRECICE_ASSERT(_kernelEval.size() == 0);

  if (this->hasConstraint(Mapping::CONSERVATIVE)) { //TODO: kann das hier sein?
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

  Eigen::MatrixXd u = _kernelEval * y;
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
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::solveConservativeWithCholesky(const time::Sample &inData, const Eigen::MatrixXd &choleskyA, Eigen::VectorXd &outData) const {
  const Eigen::VectorXd &linearisedVectors = inData.values;

  const size_t          n = _greedyIDs.size();
  const Eigen::MatrixXd y = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, _outSize).transpose();

  Eigen::MatrixXd greedySolution = _kernelEval * y;
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
    outData(Eigen::seqN(d, _outSize, inData.dataDims)) = _kernelEval.transpose() * interpolationCoeffs.col(d);
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
    outData(Eigen::seqN(d, _outSize, inData.dataDims)) = _kernelEval.transpose() * interpolationCoeffs.col(d);
  }
  if (_usesPolynomial) {
    for (int d = 0; d < inData.dataDims; d++) {
      outData(Eigen::seqN(d, _outSize, inData.dataDims)) += _polyMatrixU * polynomialCoeffs.col(d);
    }
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void GreedyMapping<RADIAL_BASIS_FUNCTION_T>::clear() {
  _greedyIDs.clear();
  _kernelEval = Eigen::MatrixXd();
  _inSize     = 0;
  _outSize    = 0;
}

} // namespace mapping
} // namespace precice

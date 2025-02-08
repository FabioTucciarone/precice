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

#define F_GREEDY 1
#define P_GREEDY 0

namespace precice {
namespace mapping {

template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
class GreedyCutMapping : public GreedyMapping<RADIAL_BASIS_FUNCTION_T> {

  using RadialBasisFctBaseMapping<RADIAL_BASIS_FUNCTION_T>::_basisFunction;
  using GreedyParameter = MappingConfiguration::GreedyParameter;
  using super = GreedyMapping<RADIAL_BASIS_FUNCTION_T>;

  using super::_log;
  using super::_greedyIDs;
  using super::_invCholeskyA;
  using super::_referenceResidualNorm;

public:

  GreedyCutMapping(
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
  Eigen::MatrixXd _kernelMatrix;

  void updateResidual(Eigen::MatrixXd &residual, const Eigen::MatrixXd &y);
  void addKernelMatrixColumn(const size_t greedyIndex);

  void updatePowerFunction(Eigen::VectorXd &powerFunction, const mesh::Vertex &x);

  virtual void buildInterpolationMatrices(const Eigen::MatrixXd &y, const Eigen::MatrixXd &startResidual, const size_t startIndex) override;
  virtual Eigen::MatrixXd recalculateResidual(const Eigen::MatrixXd &y, const size_t basisExtend) override;

  void solveConservative(const time::Sample &inData, Eigen::VectorXd &outData) const;
  void solveConsistent(const time::Sample &inData, Eigen::VectorXd &outData) const;
};


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
GreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::GreedyCutMapping(
    Mapping::Constraint     constraint,
    int                     dimensions,
    RADIAL_BASIS_FUNCTION_T function,
    std::array<bool, 3>     deadAxis,
    Polynomial              polynomial,
    GreedyParameter         greedyParameter)
    : GreedyMapping<RADIAL_BASIS_FUNCTION_T>(constraint, dimensions, function, deadAxis, polynomial, greedyParameter)
{ }


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::addKernelMatrixColumn(const size_t greedyIndex) {

  const mesh::Mesh::VertexContainer &inputVertices = super::_inputMesh->vertices();

  const auto &v = inputVertices.at(_greedyIDs.at(greedyIndex)).rawCoords(); // greedyIndex = n - 1 = _greedyIDs.size() - 1

  for (size_t i = 0; i < super::_inSize; i++) {
    const auto & u = inputVertices.at(i).rawCoords();
    const double d = computeSquaredDifference(u, v, super::_activeAxis);

    _kernelMatrix(i, greedyIndex) = _basisFunction.evaluate(std::sqrt(d));
  }
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::updatePowerFunction(Eigen::VectorXd &powerFunction, const mesh::Vertex &x) {

  const size_t n = _greedyIDs.size() - 1;
  for (size_t j = 0; j < super::_inSize; j++) {
    const auto &y       = super::_inputMesh->vertices().at(j).rawCoords();
    _kernelMatrix(j, n) = _basisFunction.evaluate(std::sqrt(computeSquaredDifference(y, x.rawCoords(), super::_activeAxis)));
  }
  powerFunction -= (Eigen::VectorXd)(_kernelMatrix.block(0, 0, super::_inSize, n + 1) * _invCholeskyA.block(n, 0, 1, n + 1).transpose()).array().square();
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
Eigen::MatrixXd GreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::recalculateResidual(const Eigen::MatrixXd &y, const size_t basisExtend) {

  PRECICE_ASSERT(basisExtend <= _greedyIDs.size());

  const Eigen::MatrixXd Cy = _invCholeskyA.block(0, 0, basisExtend, basisExtend).template triangularView<Eigen::Lower>() * y(_greedyIDs, Eigen::all).block(0, 0, basisExtend, y.cols());
  Eigen::MatrixXd interpolationCoeffs = _invCholeskyA.block(0, 0, basisExtend, basisExtend).transpose().template triangularView<Eigen::Upper>() * Cy;
  return y - _kernelMatrix.block(0, 0, super::_inSize, basisExtend) * interpolationCoeffs;
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::updateResidual(Eigen::MatrixXd &residual, const Eigen::MatrixXd &y) {

  PRECICE_ASSERT(!_greedyIDs.empty());

  const size_t n = _greedyIDs.size() - 1;
  const Eigen::MatrixXd cy = _invCholeskyA.block(n, 0, 1, n + 1) * y(_greedyIDs, Eigen::all); // ersetze mit <_invCholeskyA.block(n, 0, 1, n + 1), _invCholeskyA.block(n, 0, 1, n + 1)>
  residual -= (_kernelMatrix.block(0, 0, super::_inSize, n + 1) * _invCholeskyA.block(n, 0, 1, n + 1).transpose()) * cy;
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::computeMapping() {

  precice::profiling::Event e("map.greedy.computeMapping.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  super::computeMapping();
  
  _invCholeskyA = Eigen::MatrixXd::Zero(super::_basisSize, super::_basisSize);
  _kernelMatrix = Eigen::MatrixXd::Zero(super::_inSize, super::_basisSize);
  if constexpr (BETA != F_GREEDY) {
    buildInterpolationMatrices(Eigen::MatrixXd(), Eigen::MatrixXd(), 0);
  } else if (this->hasConstraint(Mapping::CONSERVATIVE)) {
    super::_nearestMapping.setMeshes(this->input(), this->output());
    super::_nearestMapping.computeMapping();
  }

  this->_hasComputedMapping = true;
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::buildInterpolationMatrices(const Eigen::MatrixXd &y, const Eigen::MatrixXd &startResidual, const size_t startIndex) {

  size_t initialSize = _greedyIDs.size();
  Eigen::VectorXd powerFunction;
  Eigen::MatrixXd residual               = startResidual;
  Eigen::VectorXd kernelVectorOldCenters = Eigen::VectorXd::Ones(super::_basisSize);
  Eigen::VectorXd basisVector            = Eigen::VectorXd::Ones(super::_basisSize);

  if constexpr (BETA != F_GREEDY) {
    powerFunction = Eigen::VectorXd(super::_inSize);
    powerFunction.fill(_basisFunction.evaluate(0));
  }
  _greedyIDs.erase(_greedyIDs.begin() + startIndex, _greedyIDs.end());

  const double kernelDiagonal = _basisFunction.evaluate(0);

  // Iterative selection of new points
  for (size_t n = startIndex; n < super::_maxIter; ++n) {

    const auto [i, greedyValue] = this->template select<BETA>(residual, powerFunction);
    const auto x                = super::_inputMesh->vertices().at(i);

    super::updateKernelVector(x, _greedyIDs, kernelVectorOldCenters);
    basisVector.head(n)  = _invCholeskyA.block(0, 0, n, n).template triangularView<Eigen::Lower>() * kernelVectorOldCenters.head(n);
    const double squareP = kernelDiagonal - basisVector.array().head(n).square().sum();
    const double invP    = 1.0 / std::sqrt(squareP);

    if (greedyValue < super::_tolerance || n == super::_basisSize - 1) {
      if (greedyValue < super::_tolerance) 
        break;
      super::calculateIncreasedNumberOfCenters();
      _kernelMatrix.conservativeResize(super::_inSize, super::_basisSize);
      _invCholeskyA.conservativeResize(super::_basisSize, super::_basisSize);
      _invCholeskyA.block(0, n + 1, super::_basisSize, super::_basisSize - n - 1) = Eigen::MatrixXd::Zero(super::_basisSize, super::_basisSize - n - 1);
      kernelVectorOldCenters.conservativeResize(super::_basisSize);
      basisVector.conservativeResize(super::_basisSize);
    }
    _greedyIDs.push_back(i);

    _invCholeskyA.block(n, 0, 1, n).noalias() = -basisVector.block(0, 0, n, 1).transpose() * _invCholeskyA.block(0, 0, n, n).template triangularView<Eigen::Lower>();
    _invCholeskyA(n, n)                       = 1;
    _invCholeskyA.block(n, 0, 1, n + 1) *= invP;

    if constexpr (BETA == F_GREEDY) {
      addKernelMatrixColumn(n);
      updateResidual(residual, y);
    } else {
      updatePowerFunction(powerFunction, x);
    }

    PRECICE_DEBUG("Iteration: {}, greedyValue = {}\n", n + 1, greedyValue);
  }
  if constexpr (BETA == F_GREEDY) {
    _referenceResidualNorm = residual.squaredNorm();
    super::printFGreedyConclusion(initialSize, startIndex);
  }
  PRECICE_INFO("Finished greedy search and construction of inverse.");

  super::fillEvaluationMatrix(startIndex);
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::solveConservative(const time::Sample &inData, Eigen::VectorXd &outData) const {

  const Eigen::VectorXd &linearisedVectors = inData.values;

  const size_t          n = super::_greedyIDs.size();
  const Eigen::MatrixXd y = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, super::_outSize).transpose();

  Eigen::MatrixXd u =  super::_kernelEval.block(0, 0, n, super::_outSize) * y;
  Eigen::MatrixXd Cu = _invCholeskyA.block(0, 0, n, n).template triangularView<Eigen::Lower>() * u;
  Eigen::MatrixXd greedySolution = (_invCholeskyA.block(0, 0, n, n).transpose().template triangularView<Eigen::Upper>() * Cu);

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
void GreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::solveConsistent(const time::Sample &inData, Eigen::VectorXd &outData) const {

  const Eigen::VectorXd &linearisedVectors = inData.values;

  const size_t    n = super::_greedyIDs.size();
  Eigen::MatrixXd y = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, super::_inSize).transpose();
  Eigen::MatrixXd polynomialCoeffs;

  if (super::_usesPolynomial) {
    polynomialCoeffs = super::_qrDecomposedQ.solve(y);
    y -= super::_polyMatrixQ * polynomialCoeffs;
  }

  const Eigen::MatrixXd z = y(super::_greedyIDs, Eigen::all);
  const Eigen::MatrixXd Cz = _invCholeskyA.block(0, 0, n, n).template triangularView<Eigen::Lower>() * z;
  const Eigen::MatrixXd interpolationCoeffs = _invCholeskyA.block(0, 0, n, n).transpose().template triangularView<Eigen::Upper>() * Cz;

  for (int d = 0; d < inData.dataDims; d++) {
    outData(Eigen::seqN(d, super::_outSize, inData.dataDims)) = super::_kernelEval.block(0, 0, n, super::_outSize).transpose() * interpolationCoeffs.col(d);
  }
  if (super::_usesPolynomial) {
    for (int d = 0; d < inData.dataDims; d++) {
      outData(Eigen::seqN(d, super::_outSize, inData.dataDims)) += super::_polyMatrixU * polynomialCoeffs.col(d);
    }
  }
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) {

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
void GreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) {

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
std::string GreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::getName() const {
  return "global-greedy RBF (f-cut-cpu-executor)";
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void GreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::clear() {
  super::clear();
  _kernelMatrix = Eigen::MatrixXd();
}

} // namespace mapping
} // namespace precice

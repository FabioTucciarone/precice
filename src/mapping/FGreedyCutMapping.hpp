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
class FGreedyCutMapping : public GreedyMapping<RADIAL_BASIS_FUNCTION_T> {

  using RadialBasisFctBaseMapping<RADIAL_BASIS_FUNCTION_T>::_basisFunction;
  using GreedyParameter = MappingConfiguration::GreedyParameter;
  using super = GreedyMapping<RADIAL_BASIS_FUNCTION_T>;

  using super::_log;
  using super::_greedyIDs;
  using super::_invCholeskyA;
  using super::_referenceResidualNorm;

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
  Eigen::MatrixXd _kernelMatrix;
  Eigen::VectorXd _powerFunction;

  void updateResidual(Eigen::MatrixXd &residual, const Eigen::MatrixXd &y);
  void addKernelMatrixColumn(const size_t greedyIndex);

  void updatePowerFunction(Eigen::VectorXd &powerFunction, const mesh::Vertex &x);

  virtual void buildInterpolationMatrices(const Eigen::MatrixXd &y, const Eigen::MatrixXd &r0, const size_t n0) override;
  virtual Eigen::MatrixXd recalculateResidual(const Eigen::MatrixXd &y, const size_t basisExtend) override;
};


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::FGreedyCutMapping(
    Mapping::Constraint     constraint,
    int                     dimensions,
    RADIAL_BASIS_FUNCTION_T function,
    std::array<bool, 3>     deadAxis,
    Polynomial              polynomial,
    GreedyParameter         greedyParameter)
    : GreedyMapping<RADIAL_BASIS_FUNCTION_T>(constraint, dimensions, function, deadAxis, polynomial, greedyParameter)
{ }


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::addKernelMatrixColumn(const size_t greedyIndex) {

  const mesh::Mesh::VertexContainer &inputVertices = super::_inputMesh->vertices();

  const auto &v = inputVertices.at(_greedyIDs.at(greedyIndex)).rawCoords(); // greedyIndex = n - 1 = _greedyIDs.size() - 1

  for (size_t i = 0; i < super::_inSize; i++) {
    const auto & u = inputVertices.at(i).rawCoords();
    const double d = computeSquaredDifference(u, v, super::_activeAxis);

    _kernelMatrix(i, greedyIndex) = _basisFunction.evaluate(std::sqrt(d));
  }
}


template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::updatePowerFunction(Eigen::VectorXd &powerFunction, const mesh::Vertex &x) {

  const size_t n = _greedyIDs.size() - 1;
  for (size_t j = 0; j < super::_inSize; j++) {
    const auto &y       = super::_inputMesh->vertices().at(j).rawCoords();
    _kernelMatrix(j, n) = _basisFunction.evaluate(std::sqrt(computeSquaredDifference(y, x.rawCoords(), super::_activeAxis)));
  }
  powerFunction -= (Eigen::VectorXd)(_kernelMatrix.block(0, 0, super::_inSize, n + 1) * _invCholeskyA.block(n, 0, 1, n + 1).transpose()).array().square();
}


// basisExtend = rebuild Index: nr Centers to use = n0 ## ## with Coeffs teste für basisExtend == 0
template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
Eigen::MatrixXd FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::recalculateResidual(const Eigen::MatrixXd &y, const size_t basisExtend) {
  PRECICE_ASSERT(basisExtend <= _greedyIDs.size());

  const Eigen::MatrixXd Cy = _invCholeskyA.block(0, 0, basisExtend, basisExtend).template triangularView<Eigen::Lower>() * y(_greedyIDs, Eigen::all).block(0, 0, basisExtend, y.cols());
  Eigen::MatrixXd interpolationCoeffs = _invCholeskyA.block(0, 0, basisExtend, basisExtend).transpose().template triangularView<Eigen::Upper>() * Cy;
  return y - _kernelMatrix.block(0, 0, super::_inSize, basisExtend) * interpolationCoeffs;
}


// n0 = nr Greedy Centers to use - 1 ## ## updateResidualAndCoeffs
template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::updateResidual(Eigen::MatrixXd &residual, const Eigen::MatrixXd &y) {
  PRECICE_ASSERT(!_greedyIDs.empty());

  const size_t n = _greedyIDs.size() - 1;
  const Eigen::MatrixXd cy = _invCholeskyA.block(n, 0, 1, n + 1) * y(_greedyIDs, Eigen::all); // ersetze mit <_invCholeskyA.block(n, 0, 1, n + 1), _invCholeskyA.block(n, 0, 1, n + 1)>
  residual -= (_kernelMatrix.block(0, 0, super::_inSize, n + 1) * _invCholeskyA.block(n, 0, 1, n + 1).transpose()) * cy;
}

template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::computeMapping() {

  precice::profiling::Event e("map.f-greedy.computeMapping.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  super::computeMapping();
  
  _invCholeskyA = Eigen::MatrixXd::Zero(super::_basisSize, super::_basisSize);
  _kernelMatrix = Eigen::MatrixXd::Zero(super::_inSize, super::_basisSize);
  if constexpr (BETA != F_GREEDY) {
    buildInterpolationMatrices(Eigen::MatrixXd(), Eigen::MatrixXd(), 0);
  }

  this->_hasComputedMapping = true;
}

template <typename RADIAL_BASIS_FUNCTION_T, int BETA> // beginne erneut bei Index n0 (= behalte n0 Zentren)
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::buildInterpolationMatrices(const Eigen::MatrixXd &y, const Eigen::MatrixXd &r0, const size_t startIndex) {

  size_t initialSize = _greedyIDs.size();
  Eigen::VectorXd powerFunction;
  Eigen::MatrixXd residual               = r0;
  Eigen::VectorXd kernelVectorOldCenters = Eigen::VectorXd::Ones(super::_basisSize);
  Eigen::VectorXd basisVector            = Eigen::VectorXd::Ones(super::_basisSize); // max(super::_basisSize, initialN)

  if constexpr (BETA != F_GREEDY) {
    powerFunction = Eigen::VectorXd(super::_inSize);
    powerFunction.fill(_basisFunction.evaluate(0));
  }
  _greedyIDs.erase(_greedyIDs.begin() + startIndex, _greedyIDs.end()); // n0 = 0 => Recalc everything

  const double kernelDiagonal = _basisFunction.evaluate(0);

  // Iterative selection of new points
  for (size_t n = startIndex; n < super::_maxIter; ++n) {

    const auto [i, fMax] = this->template select<BETA>(residual, powerFunction);
    const auto x         = super::_inputMesh->vertices().at(i);

    super::updateKernelVector(x, _greedyIDs, kernelVectorOldCenters);
    basisVector.head(n)  = _invCholeskyA.block(0, 0, n, n).template triangularView<Eigen::Lower>() * kernelVectorOldCenters.head(n);
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

    PRECICE_DEBUG("Iteration: {}, fMax = {}\n", n + 1, fMax, squareP);
  }
  if constexpr (BETA == F_GREEDY) {
    _referenceResidualNorm = residual.squaredNorm();
    super::printFGreedyConclusion(initialSize, startIndex);
  }
  PRECICE_INFO("Finished greedy search and construction of inverse.");

  super::fillEvaluationMatrix(startIndex);
}

template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) {
  precice::profiling::Event mapConsistentEvent("map.greedy.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);
  if constexpr (BETA == F_GREEDY) {
    super::solveConsistentFGreedy(inData, outData);
  } else {
    super::solveConsistentWithCut(inData, _invCholeskyA, outData);
  }
  mapConsistentEvent.addData("basisSize", super::_greedyIDs.size());
  mapConsistentEvent.addData("inSize", super::_inSize);
}

template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) {
  precice::profiling::Event solveEvent("map.greedy.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  const Eigen::VectorXd &linearisedVectors = inData.values;
  Eigen::MatrixXd y = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, super::_inSize).transpose();

  if constexpr (BETA == F_GREEDY) {
    super::updateInterpolationMatrices(y);
  }
  super::solveConservativeWithCut(inData, _invCholeskyA, outData);
}

template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
std::string FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::getName() const {
  return "global-greedy RBF (f-cut-cpu-executor)";
}

template <typename RADIAL_BASIS_FUNCTION_T, int BETA>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T, BETA>::clear() {
  super::clear();
  _kernelMatrix = Eigen::MatrixXd();
}

} // namespace mapping
} // namespace precice

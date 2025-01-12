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

  void updateResidual(const Eigen::MatrixXd &y, Eigen::MatrixXd &residual);
  void addKernelMatrixColumn(const size_t greedyIndex);

  virtual void buildInterpolationMatrices(const Eigen::MatrixXd &y, const Eigen::MatrixXd &r0, const size_t n0) override;
  virtual Eigen::MatrixXd recalculateResidual(const Eigen::MatrixXd &y, const size_t basisExtend) override;
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
Eigen::MatrixXd FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::recalculateResidual(const Eigen::MatrixXd &y, const size_t basisExtend) {
  PRECICE_ASSERT(basisExtend <= _greedyIDs.size());

  const Eigen::MatrixXd Cy = _invCholeskyA.block(0, 0, basisExtend, basisExtend).template triangularView<Eigen::Lower>() * y(_greedyIDs, Eigen::all).block(0, 0, basisExtend, y.cols());
  Eigen::MatrixXd interpolationCoeffs = _invCholeskyA.block(0, 0, basisExtend, basisExtend).transpose().template triangularView<Eigen::Upper>() * Cy;
  return y - _kernelMatrix.block(0, 0, super::_inSize, basisExtend) * interpolationCoeffs;
}


// n0 = nr Greedy Centers to use - 1 ## ## updateResidualAndCoeffs
template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::updateResidual(const Eigen::MatrixXd &y, Eigen::MatrixXd &residual) {
  PRECICE_ASSERT(!_greedyIDs.empty());

  const size_t n = _greedyIDs.size() - 1;
  const Eigen::MatrixXd cy = _invCholeskyA.block(n, 0, 1, n + 1) * y(_greedyIDs, Eigen::all);
  residual -= (_kernelMatrix.block(0, 0, super::_inSize, n + 1) * _invCholeskyA.block(n, 0, 1, n + 1).transpose()) * cy;
  //_interpolationCoeffs.block(0, 0, n + 1, y.cols()) += _invCholeskyA.block(n, 0, 1, n + 1).transpose() * cy;
  //residual = y - (_kernelMatrix.block(0, 0, super::_inSize, n + 1) * _interpolationCoeffs.block(0, 0, n + 1, y.cols()));
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::computeMapping() {

  precice::profiling::Event e("map.f-greedy.computeMapping.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  super::computeMapping();
  _invCholeskyA = Eigen::MatrixXd::Zero(super::_basisSize, super::_basisSize);
  _kernelMatrix = Eigen::MatrixXd::Zero(super::_inSize, super::_basisSize);
  this->_hasComputedMapping = true;
}

template <typename RADIAL_BASIS_FUNCTION_T> // beginne erneut bei Index n0 (= behalte n0 Zentren)
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::buildInterpolationMatrices(const Eigen::MatrixXd &y, const Eigen::MatrixXd &r0, const size_t startIndex) {

  Eigen::MatrixXd residual               = r0;
  Eigen::VectorXd kernelVectorOldCenters = Eigen::VectorXd::Ones(super::_basisSize);
  Eigen::VectorXd basisVector            = Eigen::VectorXd::Ones(super::_basisSize); // max(super::_basisSize, initialN)

   size_t initialSize = _greedyIDs.size();
  _greedyIDs.erase(_greedyIDs.begin() + startIndex, _greedyIDs.end()); // n0 = 0 => Recalc everything

  const double kernelDiagonal = _basisFunction.evaluate(0);

  // Iterative selection of new points
  for (size_t n = startIndex; n < super::_maxIter; ++n) {

    const auto [i, fMax] = super::select(residual);
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

    addKernelMatrixColumn(n);
    updateResidual(y, residual);

    PRECICE_DEBUG("Iteration: {}, fMax = {}\n", n + 1, fMax, squareP);
  }
  _referenceResidualNorm = residual.squaredNorm();
  super::printFGreedyConclusion(initialSize, startIndex);

  PRECICE_INFO("Finished greedy search and construction of inverse.");

  super::fillEvaluationMatrix(startIndex);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) {
  super::solveConsistentFGreedy(inData, outData);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) {

  precice::profiling::Event e("map.f-greedy.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  const Eigen::VectorXd &linearisedVectors = inData.values;
  Eigen::MatrixXd y = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, super::_inSize).transpose();
  super::updateInterpolationMatrices(y);
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
}

} // namespace mapping
} // namespace precice

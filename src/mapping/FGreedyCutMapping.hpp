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
  using GreedyMapping<RADIAL_BASIS_FUNCTION_T>::_log;
  using GreedyParameter = MappingConfiguration::GreedyParameter;
  using super = GreedyMapping<RADIAL_BASIS_FUNCTION_T>;

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
  Eigen::MatrixXd _cut;

  void recalculateResidual(const Eigen::MatrixXd &inputData, Eigen::MatrixXd &interpolationCoeffs, Eigen::MatrixXd &residual);
  Eigen::MatrixXd buildInterpolationMatrices(const Eigen::MatrixXd &inputData);
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
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::recalculateResidual(const Eigen::MatrixXd &inputData, Eigen::MatrixXd &interpolationCoeffs, Eigen::MatrixXd &residual) {

  //TODO Nutze aus, dass Kernel Matrix Symmetrisch ist?

  const mesh::Mesh::VertexContainer &inputVertices = super::_inputMesh->vertices();

  const int   n = super::_greedyIDs.size();
  const auto &v = inputVertices.at(super::_greedyIDs.at(n - 1)).rawCoords();

  for (size_t i = 0; i < super::_inSize; i++) {
    const auto & u          = inputVertices.at(i).rawCoords();
    const double d          = computeSquaredDifference(u, v, super::_activeAxis);
    _kernelMatrix(i, n - 1) = _basisFunction.evaluate(std::sqrt(d));
  }
  const Eigen::MatrixXd cy = _cut.block(n - 1, 0, 1, n) * inputData.transpose()(Eigen::all, super::_greedyIDs).transpose();
  interpolationCoeffs.block(0, 0, n, inputData.cols()) += _cut.block(n - 1, 0, 1, n).transpose() * cy;
  // residual = (inputData - (_kernelMatrix(Eigen::all, super::_greedyIDs) * interpolationCoeffs.block(0, 0, n, inputData.rows())).transpose()).cwiseAbs(); //TODO: Segmentation Fault
  residual = (inputData - (_kernelMatrix.block(0, 0, super::_inSize, n) * interpolationCoeffs.block(0, 0, n, inputData.cols()))).cwiseAbs();
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::computeMapping() {

  precice::profiling::Event e("map.f-greedy-cut.computeMapping.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  super::computeMapping();
  _cut          = Eigen::MatrixXd::Zero(super::_basisSize, super::_basisSize);
  _kernelMatrix = Eigen::MatrixXd::Zero(super::_inSize, super::_basisSize);

  /* const mesh::Mesh::VertexContainer &inputVertices = super::_inputMesh->vertices();
  for (size_t j = 0; j < super::_basisSize; j++) { 
    for (size_t i = 0; i < super::_inSize; i++) {
      const auto & u = inputVertices.at(i).rawCoords();
      const auto & v = inputVertices.at(j).rawCoords();
      const double d = computeSquaredDifference(u, v, super::_activeAxis);
      _kernelMatrix(i, j) = _basisFunction.evaluate(std::sqrt(d));
    }
  } */
  this->_hasComputedMapping = true;
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::MatrixXd FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::buildInterpolationMatrices(const Eigen::MatrixXd &inputData) {

  Eigen::MatrixXd residual               = inputData;
  Eigen::VectorXd kernelVectorOldCenters = Eigen::VectorXd::Ones(super::_basisSize);
  Eigen::VectorXd basisVector            = Eigen::VectorXd::Ones(super::_basisSize);
  Eigen::MatrixXd interpolationCoeffs    = Eigen::MatrixXd::Zero(super::_basisSize, inputData.cols());
  super::_greedyIDs.clear();

  const double kernelDiagonal = _basisFunction.evaluate(0);

  // Iterative selection of new points
  for (size_t n = 0; n < super::_maxIter; ++n) {

    const auto [i, fMax] = super::select(residual);
    const auto x         = super::_inputMesh->vertices().at(i);

    super::updateKernelVector(x, super::_greedyIDs, kernelVectorOldCenters);
    basisVector.head(n)  = _cut.block(0, 0, n, n).triangularView<Eigen::Lower>() * kernelVectorOldCenters.head(n);
    const double squareP = kernelDiagonal - basisVector.array().head(n).square().sum();
    const double invP    = 1.0 / std::sqrt(squareP);

    if (fMax < super::_tolerance || n == super::_basisSize - 1) {
      if (fMax < super::_tolerance) 
        break;
      super::calculateIncreasedNumberOfCenters();
      _kernelMatrix.conservativeResize(super::_inSize, super::_basisSize);
      _cut.conservativeResize(super::_basisSize, super::_basisSize);
      _cut.block(0, n + 1, super::_basisSize, super::_basisSize - n - 1) = Eigen::MatrixXd::Zero(super::_basisSize, super::_basisSize - n - 1);
      kernelVectorOldCenters.conservativeResize(super::_basisSize);
      basisVector.conservativeResize(super::_basisSize);
      interpolationCoeffs.conservativeResize(super::_basisSize, inputData.cols());
      interpolationCoeffs.block(n + 1, 0, super::_basisSize - n - 1, inputData.cols()) = Eigen::MatrixXd::Zero(super::_basisSize - n - 1, inputData.cols());
    }
    super::_greedyIDs.push_back(i);

    _cut.block(n, 0, 1, n).noalias() = -basisVector.block(0, 0, n, 1).transpose() * _cut.block(0, 0, n, n).triangularView<Eigen::Lower>();
    _cut(n, n)                       = 1;
    _cut.block(n, 0, 1, n + 1) *= invP;

    recalculateResidual(inputData, interpolationCoeffs, residual);

    PRECICE_DEBUG("Iteration: {}, fMax = {}\n", n + 1, fMax, squareP);
  }

  PRECICE_INFO("Finished greedy search and construction of inverse.");

  super::fillEvaluationMatrix(0);
  return interpolationCoeffs;
}


template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) {
  
  precice::profiling::Event e("map.f-greedy-cut.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  const Eigen::VectorXd &linearisedVectors = inData.values;

  Eigen::MatrixXd y = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, super::_inSize).transpose();

  if (super::_usesPolynomial) {
    super::fillPolynomialMatrices();
    const Eigen::MatrixXd polynomialCoeffs = super::_qrDecomposedQ.solve(y);
    y -= super::_polyMatrixQ * polynomialCoeffs;

    buildInterpolationMatrices(y);
    const size_t n = super::_greedyIDs.size();

    const Eigen::MatrixXd Cy = _cut.block(0, 0, n, n).triangularView<Eigen::Lower>() * y(super::_greedyIDs, Eigen::all);
    const Eigen::MatrixXd interpolationCoeffs = _cut.block(0, 0, n, n).transpose().triangularView<Eigen::Upper>() * Cy;
    
    for (int d = 0; d < inData.dataDims; d++) {
      outData(Eigen::seqN(d, super::_outSize, inData.dataDims)) = super::_kernelEval.transpose() * interpolationCoeffs.col(d) + super::_polyMatrixU * polynomialCoeffs.col(d);
    }
  } else {
    const Eigen::MatrixXd interpolationCoeffs = buildInterpolationMatrices(y);
    const size_t n = super::_greedyIDs.size();
    for (int d = 0; d < inData.dataDims; d++) {
      outData(Eigen::seqN(d, super::_outSize, inData.dataDims)) = super::_kernelEval.transpose() * interpolationCoeffs.col(d).head(n);
    }
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) {

  precice::profiling::Event e("map.f-greedy-cut.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  const Eigen::VectorXd &linearisedVectors = inData.values;
  Eigen::MatrixXd inputData = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, super::_inSize).transpose();
  buildInterpolationMatrices(inputData);
  super::solveConservativeWithCut(inData, _cut, outData);
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::string FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::getName() const {
  return "global-greedy RBF (f-cut-cpu-executor)";
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::clear() {
  super::clear();
  _kernelMatrix = Eigen::MatrixXd();
  _cut          = Eigen::MatrixXd();
}


} // namespace mapping
} // namespace precice

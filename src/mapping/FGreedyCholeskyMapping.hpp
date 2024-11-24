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
class FGreedyCholeskyMapping : public GreedyMapping<RADIAL_BASIS_FUNCTION_T> {

  using RadialBasisFctBaseMapping<RADIAL_BASIS_FUNCTION_T>::_basisFunction;
  using GreedyMapping<RADIAL_BASIS_FUNCTION_T>::_log;
  using GreedyParameter = MappingConfiguration::GreedyParameter;
  using super = GreedyMapping<RADIAL_BASIS_FUNCTION_T>;

public:

  FGreedyCholeskyMapping(
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

  void buildInterpolationMatrices(const Eigen::MatrixXd &inputData);
};


template <typename RADIAL_BASIS_FUNCTION_T>
FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::FGreedyCholeskyMapping(
    Mapping::Constraint     constraint,
    int                     dimensions,
    RADIAL_BASIS_FUNCTION_T function,
    std::array<bool, 3>     deadAxis,
    Polynomial              polynomial,
    GreedyParameter         greedyParameter)
    : GreedyMapping<RADIAL_BASIS_FUNCTION_T>(constraint, dimensions, function, deadAxis, polynomial, greedyParameter)
{ }


template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::computeMapping() {

  precice::profiling::Event e("map.f-greedy-cholesky.computeMapping.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  super::computeMapping();
  _basisMatrix.resize(super::_inSize, super::_basisSize);

  this->_hasComputedMapping = true;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::buildInterpolationMatrices(const Eigen::MatrixXd &y) {

  Eigen::VectorXd basisVector(super::_inSize);
  Eigen::MatrixXd residual = y;
  super::_greedyIDs.clear();

  // Iterative selection of new points
  for (size_t n = 0; n < super::_maxIter; ++n) {

    const auto [i, fMax] = super::select(residual);
    const auto x         = super::_inputMesh->vertices().at(i);

    super::updateKernelVector(x, boost::irange(0UL, super::_inSize), basisVector);
    basisVector -= _basisMatrix.block(0, 0, super::_inSize, n) * _basisMatrix.block(i, 0, 1, n).transpose();

    if (fMax < super::_tolerance || basisVector(i) <= 0 || n == super::_basisSize - 1) {
      if (fMax < super::_tolerance || basisVector(i) <= 0) {
        std::cout << " >> STOP " << fMax << " < " << super::_tolerance << std::endl;
        break;
      }
      super::calculateIncreasedNumberOfCenters();
      _basisMatrix.conservativeResize(super::_inSize, super::_basisSize);
    }
    super::_greedyIDs.push_back(i);

    const double invP = 1.0 / std::sqrt(basisVector(i));
    basisVector *= invP;
    _basisMatrix.col(n) = basisVector;

    const Eigen::RowVectorXd newtonCoefficient = residual.row(i) * invP;
    residual -= basisVector * newtonCoefficient;

    PRECICE_DEBUG("Iteration: {}, fMax = {}\n", n + 1, fMax);
  }

  PRECICE_INFO("Finished greedy search. Reordering cholesky matrix.");

  std::cout << " >> INITIAL RESIDUAL: " << residual.squaredNorm() << std::endl;

  _choleskyA = _basisMatrix(super::_greedyIDs, Eigen::seqN(0, super::_greedyIDs.size()));

  super::fillEvaluationMatrix();
}


template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) {
  
  precice::profiling::Event e("map.f-greedy-cholesky.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  const Eigen::VectorXd &linearisedVectors = inData.values;
  Eigen::MatrixXd y = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, super::_inSize).transpose();

  Eigen::MatrixXd polynomialCoeffs;
  if (super::_usesPolynomial) {
    super::fillPolynomialMatrices();
    polynomialCoeffs = super::_qrDecomposedQ.solve(y);
    y -= super::_polyMatrixQ * polynomialCoeffs;
  }

  if (super::_greedyIDs.size() == 0) {
    std::cout << " >> BUILD " << std::endl;
    buildInterpolationMatrices(y);
  }
  
  precice::profiling::Event er("recalcResidual", profiling::Synchronize);

  Eigen::MatrixXd residual = y;
  for (size_t i = 0; i < super::_greedyIDs.size(); i++) {
    const double invP = 1.0 / _choleskyA(i, i);
    const Eigen::VectorXd newtonCoefficient = residual.row(super::_greedyIDs.at(i)).transpose() * invP;
    residual -= _basisMatrix.col(i) * newtonCoefficient.transpose();
  }
  er.stop();
  std::cout << " >> RESIDUAL: " << residual.squaredNorm() << std::endl;

  Eigen::MatrixXd interpolationCoeffs = y(super::_greedyIDs, Eigen::all);
  _choleskyA.triangularView<Eigen::Lower>().solveInPlace(interpolationCoeffs);
  _choleskyA.transpose().triangularView<Eigen::Upper>().solveInPlace(interpolationCoeffs);

  for (int d = 0; d < inData.dataDims; d++) {
    outData(Eigen::seqN(d, super::_outSize, inData.dataDims)) = super::_kernelEval.transpose() * interpolationCoeffs.col(d);
  }
  if (super::_usesPolynomial) {
    for (int d = 0; d < inData.dataDims; d++) {
      outData(Eigen::seqN(d, super::_outSize, inData.dataDims)) += super::_polyMatrixU * polynomialCoeffs.col(d);
    }
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) {

  precice::profiling::Event e("map.f-greedy-cholesky.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  const Eigen::VectorXd &linearisedVectors = inData.values;
  Eigen::MatrixXd inputData = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, super::_inSize).transpose();
  buildInterpolationMatrices(inputData);
  super::solveConservativeWithCholesky(inData, _choleskyA, outData);
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::string FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::getName() const {
  return "global-greedy RBF (f-cholesky-cpu-executor)";
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::clear() {
  super::clear();
  _choleskyA   = Eigen::MatrixXd();
  _basisMatrix = Eigen::MatrixXd();
}

} // namespace mapping
} // namespace precice

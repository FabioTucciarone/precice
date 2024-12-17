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


#define PRINT_FULL_OUTPUT false


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

  double _referenceResidualNorm;

  std::pair<int, double> selectMax(const Eigen::MatrixXd &residual) const;
  std::pair<int, double> selectMax(const Eigen::MatrixXd &residual, const std::vector<int> &greedyIDs) const;
  std::pair<int, double> selectMin(const Eigen::MatrixXd &residual) const;

  void buildInterpolationMatrices(const Eigen::MatrixXd &y);
  void buildInterpolationMatrices(const Eigen::MatrixXd &residual0, const size_t n0);

  void exchange(const Eigen::MatrixXd &y, size_t removealN);
  void reorderBasis(const Eigen::MatrixXd &y, const size_t removealN);

  Eigen::MatrixXd recalculateResidual(const Eigen::MatrixXd &y, size_t basisExtend);
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
std::pair<int, double> FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::selectMax(const Eigen::MatrixXd &residual) const {
  Eigen::Index maxIndex;
  double       maxValue = residual.rowwise().squaredNorm().maxCoeff(&maxIndex);
  return {maxIndex, maxValue};
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::pair<int, double> FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::selectMax(const Eigen::MatrixXd &residual, const std::vector<int> &greedyIDs) const {
  Eigen::Index maxIndex;
  double       maxValue = residual.rowwise().squaredNorm()(greedyIDs).maxCoeff(&maxIndex);
  return {maxIndex, maxValue};
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::pair<int, double> FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::selectMin(const Eigen::MatrixXd &residual) const {
  Eigen::Index minIndex;
  double       minValue = residual.rowwise().squaredNorm()(super::_greedyIDs).minCoeff(&minIndex);
  return {minIndex, minValue};
}

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

    const auto [i, fMax] = selectMax(residual);
    const auto x         = super::_inputMesh->vertices().at(i);

    super::updateKernelVector(x, boost::irange(0UL, super::_inSize), basisVector);
    basisVector -= _basisMatrix.block(0, 0, super::_inSize, n) * _basisMatrix.block(i, 0, 1, n).transpose();

    if (fMax < super::_tolerance || basisVector(i) <= 0 || n == super::_basisSize - 1) {
      if (fMax < super::_tolerance || basisVector(i) <= 0) {
        std::cout << " > STOP " << fMax << " < " << super::_tolerance << std::endl;
        break;
      }
      super::calculateIncreasedNumberOfCenters();
      _basisMatrix.conservativeResize(super::_inSize, super::_basisSize);
    }
    super::_greedyIDs.push_back(i);

    const double invP = 1.0 / std::sqrt(basisVector(i));
    basisVector *= invP;
    _basisMatrix.col(n) = basisVector;

    const Eigen::RowVectorXd newtonCoefficient = residual.row(i) * invP; // temp alloc
    residual -= basisVector * newtonCoefficient;

    PRECICE_DEBUG("Iteration: {}, fMax = {}\n", n + 1, fMax);
  }

  _choleskyA = _basisMatrix(super::_greedyIDs, Eigen::seqN(0, super::_greedyIDs.size()));
  _referenceResidualNorm = residual.squaredNorm();

  super::fillEvaluationMatrix();
}


template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::buildInterpolationMatrices(const Eigen::MatrixXd &r0, const size_t n0) {

  precice::profiling::Event exchangeEvent("buildInterpolationMatrices(r0,n0)", profiling::Synchronize);

  std::cout << "\n > buildInterpolationMatrices(const Eigen::MatrixXd &r0, const size_t n0): \n";

  Eigen::VectorXd basisVector(super::_inSize);
  Eigen::MatrixXd residual = r0; // KOPIE???

  size_t initialN = super::_greedyIDs.size();

  super::_greedyIDs.erase(super::_greedyIDs.begin() + n0, super::_greedyIDs.end()); // n0 = 0 => Recalc everything
  // Iterative selection of new points
  for (size_t n = n0; n < super::_maxIter; ++n) {

    const auto [i, fMax] = selectMax(residual);
    const auto x         = super::_inputMesh->vertices().at(i);

    super::updateKernelVector(x, boost::irange(0UL, super::_inSize), basisVector);
    basisVector -= _basisMatrix.block(0, 0, super::_inSize, n) * _basisMatrix.block(i, 0, 1, n).transpose();

    if (fMax < super::_tolerance || basisVector(i) <= 0 || n == super::_basisSize - 1) {
      if (fMax < super::_tolerance || basisVector(i) <= 0) {
        // std::cout << " > STOP " << fMax << " < " << super::_tolerance << std::endl;
        break;
      }
      super::calculateIncreasedNumberOfCenters();
      _basisMatrix.conservativeResize(super::_inSize, super::_basisSize);
    }
    super::_greedyIDs.push_back(i);

    const double invP = 1.0 / std::sqrt(basisVector(i));
    basisVector *= invP;
    _basisMatrix.col(n) = basisVector;

    const Eigen::RowVectorXd newtonCoefficient = residual.row(i) * invP; // temp alloc
    residual -= basisVector * newtonCoefficient;
  }

  //PRECICE_INFO("Finished greedy search. Reordering cholesky matrix.");

  _choleskyA = _basisMatrix(super::_greedyIDs, Eigen::seqN(0, super::_greedyIDs.size()));

  _referenceResidualNorm = residual.squaredNorm();
  int difference = int(initialN) - int(super::_greedyIDs.size());
  fmt::print("   [{}] {} -> {} -> {} (removed: {}%, difference: {})\n", super::_inSize, initialN, n0, super::_greedyIDs.size(), float(initialN - n0) / initialN * 100, difference);

  super::fillEvaluationMatrix();
}


template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::exchange(const Eigen::MatrixXd &y, size_t removalN) {

  precice::profiling::Event removeEvent("exchange(y, removalN) > Remove", profiling::Synchronize);

  size_t n = super::_greedyIDs.size();
  Eigen::MatrixXd inverseCholeskyA = Eigen::MatrixXd::Identity(n, n);
  _choleskyA.triangularView<Eigen::Lower>().solveInPlace(inverseCholeskyA);
  Eigen::MatrixXd interpolationCoeffs = inverseCholeskyA.transpose() * inverseCholeskyA * y(super::_greedyIDs, Eigen::all);
  
  Eigen::MatrixXd _partialInverseA = Eigen::MatrixXd::Zero(2 * removalN - 1, 2 * removalN - 1);


  double minResidualNorm = std::numeric_limits<double>::max();
  double rebuildIndex = n; // TODO: Randfall am Ende "minM = n" STARTWERT????

  size_t blockHeight = removalN;
  for (size_t m = 0; m < n; m += blockHeight) { // m in greedy Raum
    if (m + 2 * removalN > n) blockHeight = n - m;
    size_t blockLength = m + blockHeight;

    _partialInverseA.block(0, 0, blockHeight, blockHeight) = inverseCholeskyA.block(m, 0, blockHeight, blockLength) * inverseCholeskyA.block(m, 0, blockHeight, blockLength).transpose();
    double partialResidual = (_partialInverseA.block(0, 0, blockHeight, blockHeight).inverse() * interpolationCoeffs.block(m, 0, blockHeight, interpolationCoeffs.cols())).squaredNorm();

    if (partialResidual <= minResidualNorm) { // < oder <= ???
      minResidualNorm = partialResidual;
      rebuildIndex = m;
    }
  }

  removeEvent.stop();

  precice::profiling::Event exchangeRebuildEvent("exchange(y, removalN) > Rebuild", profiling::Synchronize);

  if (rebuildIndex != n) {
    buildInterpolationMatrices(recalculateResidual(y, rebuildIndex), rebuildIndex);
  }

  exchangeRebuildEvent.stop();
}

// REORDER unreusable in GREEDY SPACE!!
template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::reorderBasis(const Eigen::MatrixXd &y, const size_t removalN) { 

  precice::profiling::Event removeEvent("reorderBasis(y, removalN) > Remove", profiling::Synchronize);

  size_t N = super::_greedyIDs.size();

  Eigen::VectorXd basisVector = Eigen::VectorXd::Zero(N); // trotzedm alle Auswertungen benötigt
  Eigen::MatrixXd residual = y(super::_greedyIDs, Eigen::all);
  Eigen::MatrixXd localBasisMatrix = Eigen::MatrixXd::Zero(N, N); 
  std::vector<int> reorderedIDs;
  reorderedIDs.reserve(N);

  for (size_t n = 0; n < N - removalN; ++n) {

    const auto [i, fMax] = selectMax(residual);
    const size_t j       = super::_greedyIDs.at(i);
    const auto   x       = super::_inputMesh->vertices().at(j);

    super::updateKernelVector(x, super::_greedyIDs, basisVector);
    basisVector -= localBasisMatrix.block(0, 0, N, n) * localBasisMatrix.block(i, 0, 1, n).transpose();

    reorderedIDs.push_back(j); // hinzufügen in globalem Raum

    const double invP = 1.0 / std::sqrt(basisVector(i));
    basisVector *= invP;
    localBasisMatrix.col(n) = basisVector;

    const Eigen::RowVectorXd newtonCoefficient = residual.row(i) * invP; // temp alloc
    residual -= basisVector * newtonCoefficient;
  }
  //PRECICE_INFO("Finished greedy search. Reordering cholesky matrix.");

  size_t rebuildIndex = N;
  for (size_t i = 0; i < N; i++) {
    if (std::find(reorderedIDs.begin(), reorderedIDs.end(), super::_greedyIDs.at(i)) == reorderedIDs.end()) { 
      rebuildIndex = i;
      break;
    }
  }

  removeEvent.stop();

  precice::profiling::Event rebuildEvent("reorderBasis(y, removalN) > Rebuild", profiling::Synchronize);

  if (rebuildIndex != N) {
    buildInterpolationMatrices(recalculateResidual(y, rebuildIndex), rebuildIndex);
  }

  rebuildEvent.stop();
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::MatrixXd FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::recalculateResidual(const Eigen::MatrixXd &y, size_t basisExtend) {
  precice::profiling::Event recalcResidualEvent("recalculateResidual(y, basisExtend)", profiling::Synchronize);

  //std::cout << "\n > recalculateResidual(const Eigen::MatrixXd &y, size_t basisExtend):\n";

  PRECICE_ASSERT(basisExtend <= super::_greedyIDs.size());

  Eigen::MatrixXd residual = y;
  for (size_t i = 0; i < basisExtend; i++) {
    int j = super::_greedyIDs.at(i);
    const double invP = 1.0 / _choleskyA(i, i); // TODO: _basisMatrix(j, i)?
    const Eigen::RowVectorXd newtonCoefficient = residual.row(j) * invP;
    residual -= _basisMatrix.col(i) * newtonCoefficient;
  }
  //std::cout << "   ||res|| = " << residual.squaredNorm() << ",  max: " << residual.maxCoeff() << "  ( basisExtend = " << basisExtend << " )\n";
  //std::cout << std::endl;

  return residual;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) {
  
  std::cout << "\nSTEP mapConsistent(inData, outData):\n";

  precice::profiling::Event mapConsistentEvent("mapConsistent(inData, outData)", profiling::Synchronize);

  const Eigen::VectorXd &linearisedVectors = inData.values;
  Eigen::MatrixXd y = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, super::_inSize).transpose();

  Eigen::MatrixXd polynomialCoeffs;
  if (super::_usesPolynomial) {
    super::fillPolynomialMatrices();
    polynomialCoeffs = super::_qrDecomposedQ.solve(y);
    y -= super::_polyMatrixQ * polynomialCoeffs;
  }

  precice::profiling::Event buildEvent("mapConsistent(inData, outData) > Build", profiling::Synchronize);

  if (super::_greedyIDs.size() == 0) {
    buildInterpolationMatrices(y, 0);
  }
  else {
    size_t n = super::_greedyIDs.size();

    enum UpdateType {REBUILD_AT_TOLERANCE, REORDER_PARTIAL_REBUILD, EXCHANGE_PARTIAL_REBUILD};
    UpdateType updateType = UpdateType::EXCHANGE_PARTIAL_REBUILD;
    double rebuildTolerance = 10 * _referenceResidualNorm;
    int removalN = 5;

    switch (updateType) {
    case UpdateType::REBUILD_AT_TOLERANCE:
      if (rebuildTolerance == 0) buildInterpolationMatrices(y, 0);
      else {
        const Eigen::MatrixXd residual = recalculateResidual(y, n);
        if (rebuildTolerance < residual.squaredNorm()) { 
          buildInterpolationMatrices(residual, 0); 
        }
      }
      break;
    case UpdateType::REORDER_PARTIAL_REBUILD:
      reorderBasis(y, removalN);
      break;
    case UpdateType::EXCHANGE_PARTIAL_REBUILD:
      exchange(y, removalN);
      break;
    }
  }

  buildEvent.stop();

  precice::profiling::Event solveEvent("mapConsistent(inData, outData) > Solve", profiling::Synchronize);

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

  solveEvent.stop();

  std::cout << "\nEND mapConsistent(inData, outData):\n\n";
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

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
class PGreedyCutMapping : public GreedyMapping<RADIAL_BASIS_FUNCTION_T> {

  using RadialBasisFctBaseMapping<RADIAL_BASIS_FUNCTION_T>::_basisFunction;
  using GreedyMapping<RADIAL_BASIS_FUNCTION_T>::_log;
  using GreedyParameter = MappingConfiguration::GreedyParameter;
  using super = GreedyMapping<RADIAL_BASIS_FUNCTION_T>;

public:

  PGreedyCutMapping(
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
  Eigen::VectorXd _powerFunction;

  void updatePowerFunction(const mesh::Vertex &x, const std::vector<int> &greedyIDs);
};


template <typename RADIAL_BASIS_FUNCTION_T>
PGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::PGreedyCutMapping(
  Mapping::Constraint     constraint,
  int                     dimensions,
  RADIAL_BASIS_FUNCTION_T function,
  std::array<bool, 3>     deadAxis,
  Polynomial              polynomial,
  GreedyParameter         greedyParameter)
    : GreedyMapping<RADIAL_BASIS_FUNCTION_T>(constraint, dimensions, function, deadAxis, polynomial, greedyParameter)
{ }

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::updatePowerFunction(const mesh::Vertex &x, const std::vector<int> &greedyIDs) {

  const size_t n = greedyIDs.size() - 1;
  for (size_t j = 0; j < super::_inSize; j++) {
    const auto &y       = super::_inputMesh->vertices().at(j).rawCoords();
    _kernelMatrix(j, n) = _basisFunction.evaluate(std::sqrt(computeSquaredDifference(y, x.rawCoords(), super::_activeAxis)));
  }
  _powerFunction -= (Eigen::VectorXd)(_kernelMatrix.block(0, 0, super::_inSize, n + 1) * _cut.block(n, 0, 1, n + 1).transpose()).array().square();
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::computeMapping() {

  precice::profiling::Event e("map.P-greedy-cut.computeMapping.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  super::computeMapping();

  _cut           = Eigen::MatrixXd::Zero(super::_basisSize, super::_basisSize);
  _powerFunction = Eigen::VectorXd(super::_inSize);
  _kernelMatrix  = Eigen::MatrixXd::Zero(super::_inSize, super::_basisSize);
  _powerFunction.fill(_basisFunction.evaluate(0));

  Eigen::VectorXd kernelVector = Eigen::VectorXd::Ones(super::_basisSize);
  Eigen::VectorXd basisVector  = Eigen::VectorXd::Ones(super::_basisSize);

  // Iterative selection of new points
  for (size_t n = 0; n < super::_maxIter; ++n) {

    auto [i, pMax] = super::select(_powerFunction);
    auto x         = super::_inputMesh->vertices().at(i);

    if (pMax < super::_tolerance || n == super::_basisSize) {
      if (pMax < super::_tolerance) 
        break;
      super::calculateIncreasedNumberOfCenters();
      _kernelMatrix.conservativeResize(super::_inSize, super::_basisSize);
      _cut.conservativeResize(super::_basisSize, super::_basisSize);
      _cut.block(0, n + 1, super::_basisSize, super::_basisSize - n - 1) = Eigen::MatrixXd::Zero(super::_basisSize, super::_basisSize - n - 1);
      kernelVector.conservativeResize(super::_basisSize);
      basisVector.conservativeResize(super::_basisSize);
      PRECICE_DEBUG("Resizing matrices.");
    }
    const double invP = 1.0 / std::sqrt(pMax);

    super::updateKernelVector(x, super::_greedyIDs, kernelVector);
    basisVector.head(n) = _cut.block(0, 0, n, n).triangularView<Eigen::Lower>() * kernelVector.head(n);

    _cut.block(n, 0, 1, n).noalias() = -basisVector.block(0, 0, n, 1).transpose() * _cut.block(0, 0, n, n).triangularView<Eigen::Lower>();
    _cut(n, n)                       = 1;
    _cut.block(n, 0, 1, n + 1) *= invP;

    super::_greedyIDs.push_back(i);
    updatePowerFunction(x, super::_greedyIDs);

    PRECICE_DEBUG("Iteration: {}, pMax = {}", n + 1, pMax);
  }
  PRECICE_INFO("Finished greedy search and construction of inverse.");

  super::fillEvaluationMatrix(0);
  if (super::_usesPolynomial) {
    super::fillPolynomialMatrices();
  }

  this->_hasComputedMapping = true;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) {
  
  precice::profiling::Event e("map.P-greedy-cut.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);
  super::solveConsistentWithCut(inData, _cut, outData);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) {

  precice::profiling::Event e("map.P-greedy-cut.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);
  super::solveConservativeWithCut(inData, _cut, outData);
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::string PGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::getName() const {
  return "global-greedy RBF (f-cut-cpu-executor)";
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::clear() {
  super::clear();
  _kernelMatrix = Eigen::MatrixXd();
  _cut          = Eigen::MatrixXd();
}

} // namespace mapping
} // namespace precice

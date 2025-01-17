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
#include "mapping/GreedyMapping.hpp"
#include "mesh/Mesh.hpp"
#include "precice/impl/Types.hpp"
#include "profiling/Event.hpp"
#include <iostream>
#include <fstream>

namespace precice {
namespace mapping {

template <typename RADIAL_BASIS_FUNCTION_T>
class PGreedyCholeskyMapping : public GreedyMapping<RADIAL_BASIS_FUNCTION_T> {

  using RadialBasisFctBaseMapping<RADIAL_BASIS_FUNCTION_T>::_basisFunction;
  using super = GreedyMapping<RADIAL_BASIS_FUNCTION_T>;
  using GreedyParameter = MappingConfiguration::GreedyParameter;

  using super::_log;
  using super::_greedyIDs;

public:

  PGreedyCholeskyMapping(
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
  Eigen::VectorXd _powerFunction;
  Eigen::MatrixXd _choleskyA;
  Eigen::MatrixXd _basisMatrix;

  std::pair<int, double> select() const;
};


template <typename RADIAL_BASIS_FUNCTION_T>
PGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::PGreedyCholeskyMapping(
  Mapping::Constraint     constraint,
  int                     dimensions,
  RADIAL_BASIS_FUNCTION_T function,
  std::array<bool, 3>     deadAxis,
  Polynomial              polynomial,
  GreedyParameter         greedyParameter)
    : GreedyMapping<RADIAL_BASIS_FUNCTION_T>(constraint, dimensions, function, deadAxis, polynomial, greedyParameter)
{ }

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::computeMapping() {
  
  precice::profiling::Event e("map.P-greedy.computeMapping.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  super::computeMapping();
  
  _basisMatrix.resize(super::_inSize, super::_basisSize);
  _powerFunction = Eigen::VectorXd(super::_inSize);
  _powerFunction.fill(_basisFunction.evaluate(0));
  Eigen::VectorXd basisVector(super::_inSize);

  PRECICE_INFO("Preallocated {}% ({}) of centers.", static_cast<size_t>((super::_basisSize / super::_inSize) * 100), super::_basisSize);

  // Iterative selection of new points
  for (size_t n = 0; n < super::_maxIter; ++n) {

    auto [i, pMax] = super::select_(_powerFunction);
    auto x         = super::_inputMesh->vertices().at(i);

    if (pMax < super::_tolerance || n == super::_basisSize) {
      if (pMax < super::_tolerance) 
        break;
      super::calculateIncreasedNumberOfCenters();
      _basisMatrix.conservativeResize(super::_inSize, super::_basisSize);
    }
    _greedyIDs.push_back(i);

    super::updateKernelVector(x, boost::irange(0UL, super::_inSize), basisVector);
    basisVector -= _basisMatrix.block(0, 0, super::_inSize, n) * _basisMatrix.block(i, 0, 1, n).transpose();
    const double invP = 1.0 / std::sqrt(pMax);
    basisVector *= invP;

    _powerFunction -= (Eigen::VectorXd) basisVector.array().square();
    _basisMatrix.col(n) = basisVector;

    PRECICE_DEBUG("Iteration: {}, pMax = {}", n + 1, pMax);
  }

  PRECICE_INFO("Finished greedy search. Reordering cholesky matrix.");

  _choleskyA   =  _basisMatrix(_greedyIDs, Eigen::seqN(0, _greedyIDs.size()));
  _basisMatrix = Eigen::MatrixXd();

  super::fillEvaluationMatrix(0);
  if (super::_usesPolynomial) {
    super::fillPolynomialMatrices();
  }

  this->_hasComputedMapping = true;
}


template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) {
  
  precice::profiling::Event mapConsistentEvent("map.P-greedy.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);
  super::solveConsistentWithCholesky(inData, _choleskyA, outData);

  mapConsistentEvent.addData("basisSize", _greedyIDs.size());
  mapConsistentEvent.addData("inSize", super::_inSize);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) {

  precice::profiling::Event e("map.P-greedy.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);
  super::solveConservativeWithCholesky(inData, _choleskyA, outData);
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::string PGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::getName() const {
  return "global-greedy RBF (P-cholesky-cpu-executor)";
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::clear(){
  super::clear();
  _choleskyA   = Eigen::MatrixXd();
  _basisMatrix = Eigen::MatrixXd();
}

} // namespace mapping
} // namespace precice

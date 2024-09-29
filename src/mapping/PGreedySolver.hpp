#pragma once

#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/irange.hpp>
#include <numeric>
#include "io/ExportVTU.hpp"
#include "mapping/config/MappingConfigurationTypes.hpp"
#include "mesh/Mesh.hpp"
#include "precice/impl/Types.hpp"
namespace precice {
namespace mapping {

/**
 * VKOGA PGreedy algorithm: reimplemented the PGreedy solver as found in
 * https://github.com/GabrieleSantin/VKOGA/blob/master/src/vkoga/pgreedy.py
 *
 * As opposed to the original example in VKOGA, our setup is slightly different in terms of when to compute what:
 *
 * Nomenclature:
 * Original: X -> our case: input mesh vertices, i.e.,
 * the vertices in two or three dimensional space on which we have data given and on which we want to build an interpolant
 * Original: Y -> our case: input data, i.e., the coupling data associated to the input mesh vertices
 * Original: X_test -> our case: output mesh vertices, i.e., the vertices (2d or 3d) on which we need to evaluate the interpolant
 * Original: Y_test -> our case: output mesh data, i.e., the unknown data values we want to evaluate, associated to the output mesh vertices
 *
 * In the original case, we typically have initially given: X and Y, such that we have two main stages:
 *
 * 1. PGreedy(params) and PGreedy.fit(X, y), which creates the reduced model
 * 2. PGreedy.predict(X_test), which evaluates the fit on the test data
 *
 * In our case, we typically have initially given: X and X_test, such that we have two (different) main stages:
 *
 * 1. PGreedy(params, X, X_test), which computes the centers and associated data structures (_cut and greedyIDs)
 * 2. PGreedy.solveConsistent(y), which evaluates the model for new data
 *
 * When an object is created, we compute the centers, the solveConsistent evaluates the center fit for new data.
 */
template <typename RADIAL_BASIS_FUNCTION_T>
class PGreedySolver {
public:
  using DecompositionType = std::conditional_t<RADIAL_BASIS_FUNCTION_T::isStrictlyPositiveDefinite(), Eigen::LLT<Eigen::MatrixXd>, Eigen::ColPivHouseholderQR<Eigen::MatrixXd>>;
  using BASIS_FUNCTION_T  = RADIAL_BASIS_FUNCTION_T;
  /// Default constructor
  PGreedySolver() = default;

  /**
   * computes the greedy centers and stores data structures to later on evaluate the reduced model
  */
  template <typename IndexContainer>
  PGreedySolver(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
                const mesh::Mesh &outputMesh, const IndexContainer &outputIDs, std::vector<bool> deadAxis, Polynomial polynomial);

  /// Maps the given input data
  Eigen::VectorXd solveConsistent(Eigen::VectorXd &inputData, Polynomial polynomial) const;

  /// Maps the given input data
  Eigen::VectorXd solveConservative(const Eigen::VectorXd &inputData, Polynomial polynomial) const;

  // Clear all stored matrices
  void clear();

  // Returns the size of the input data
  Eigen::Index getInputSize() const;

  // Returns the size of the input data
  Eigen::Index getOutputSize() const;

private:
  precice::logging::Logger _log{"mapping::PGreedySolver"};

  std::pair<int, double> selectionRule(const mesh::Mesh &inputMesh, RADIAL_BASIS_FUNCTION_T basisFunction);
  std::pair<int, double> select(const mesh::Mesh &inputMesh, RADIAL_BASIS_FUNCTION_T basisFunction);
  Eigen::VectorXd        predict(const mesh::Mesh::VertexContainer &vertices, RADIAL_BASIS_FUNCTION_T basisFunction);

  /// max iterations
  const int _maxIter = 1000;

  /// n_randon
  const double _tolP = 1e-10;

  /// the selected centers
  mesh::Mesh::VertexContainer _centers;

  /// c upper triangular
  Eigen::MatrixXd _cut;

  std::vector<int> _greedyIDs;

  Eigen::Index    _inSize  = 0;
  Eigen::Index    _outSize = 0;
  Eigen::MatrixXd _kernel_eval;

  Eigen::MatrixXd _basisMatrix;
  Eigen::VectorXd _powerFunction;
};

// ------- Non-Member Functions ---------

/// Deletes all dead directions from fullVector and returns a vector of reduced dimensionality.
inline double computeSquaredDifference2(
    const std::array<double, 3> &u,
    std::array<double, 3>        v,
    const std::array<bool, 3> &  activeAxis = {{true, true, true}})
{
  // Subtract the values and multiply out dead dimensions
  for (unsigned int d = 0; d < v.size(); ++d) {
    v[d] = (u[d] - v[d]) * static_cast<int>(activeAxis[d]);
  }
  // @todo: this can be replaced by std::hypot when moving to C++17
  return std::accumulate(v.begin(), v.end(), static_cast<double>(0.), [](auto &res, auto &val) { return res + val * val; });
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::pair<int, double> PGreedySolver<RADIAL_BASIS_FUNCTION_T>::select(const mesh::Mesh &inputMesh, RADIAL_BASIS_FUNCTION_T basisFunction)
{
  // Sample is here just our input distribution
  Eigen::Index maxIndex;
  double       maxValue = _powerFunction.maxCoeff(&maxIndex);
  //_powerFunction[maxIndex] = -std::numeric_limits<double>::infinity(); // TODO: Nicht doppelt auswählen!
  return {maxIndex, maxValue};
}


template <typename RADIAL_BASIS_FUNCTION_T, typename VertexContainer>
Eigen::MatrixXd buildEvaluationMatrix(RADIAL_BASIS_FUNCTION_T basisFunction, const VertexContainer &outputMesh, const VertexContainer &inputMesh, const std::vector<int> &greedyIDs)
{
  const mesh::Mesh::VertexContainer& inputVertices = inputMesh.vertices();
  const mesh::Mesh::VertexContainer& outputVertices = outputMesh.vertices();

  Eigen::MatrixXd matrixA(greedyIDs.size(), outputVertices.size());

  for (size_t i = 0; i < greedyIDs.size(); i++) {
    const auto &u = inputVertices.at(greedyIDs.at(i)).rawCoords();
    for (size_t j = 0; j < outputVertices.size(); j++) {
      const auto  &v                 = outputVertices.at(j).rawCoords();
      const double squaredDifference = computeSquaredDifference2(u, v, {{true, true, true}}); //TODO: Aktive Achsen
      matrixA(i, j)  = basisFunction.evaluate(std::sqrt(squaredDifference));
    }
  }

  return matrixA;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void updateKernelVector(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, Eigen::VectorXd &kernelVector, const mesh::Vertex &x)
{
  const mesh::Mesh::VertexContainer& vertices = inputMesh.vertices();
  for (size_t j = 0; j < vertices.size(); j++)
  {
    const auto &y = vertices.at(j).rawCoords();
    kernelVector(j) = basisFunction.evaluate(std::sqrt(computeSquaredDifference2(x.rawCoords(), y, {{true, true, true}}))); //TODO: Aktive Achsen
  }
}

//TODO: SOLVER

template <typename RADIAL_BASIS_FUNCTION_T>
template <typename IndexContainer>
PGreedySolver<RADIAL_BASIS_FUNCTION_T>::PGreedySolver(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
                                                      const mesh::Mesh &outputMesh, const IndexContainer &outputIDs, std::vector<bool> deadAxis, Polynomial polynomial)
{
  PRECICE_ASSERT(polynomial == Polynomial::OFF, "Poly off");
  PRECICE_ASSERT(_centers.empty());
  PRECICE_ASSERT(_greedyIDs.empty());
  PRECICE_ASSERT(_cut.size() == 0);
  PRECICE_ASSERT(_kernel_eval.size() == 0); 

  _inSize  = inputMesh.vertices().size();
  const int matWidth = std::min(static_cast<int>(_inSize), _maxIter); // maximal number of used basis functions
  _outSize = outputMesh.vertices().size();
  _powerFunction = Eigen::VectorXd(_inSize);
  _powerFunction.fill(basisFunction.evaluate(0));
  _basisMatrix = Eigen::MatrixXd::Zero(_inSize, matWidth);
  _cut = Eigen::MatrixXd::Zero(matWidth, matWidth);
  Eigen::MatrixXd _cop = Eigen::MatrixXd::Zero(matWidth, matWidth);
  Eigen::VectorXd v(_inSize);
  std::vector<bool> centerBits(_inSize);

  for(int j = 0; j < _inSize; ++j) centerBits.at(j) = false;

  // Convert dead axis vector into an active axis array so that we can handle the reduction more easily
  std::array<bool, 3> activeAxis({{false, false, false}});
  std::transform(deadAxis.begin(), deadAxis.end(), activeAxis.begin(), [](const auto ax) { return !ax; });

  // Iterative selection of new points
  for (int n = 0; n < _maxIter; ++n) {

    auto [i, pMax] = select(inputMesh, basisFunction);
    auto x = inputMesh.vertices().at(i); 

    if (pMax < _tolP) break;

    _greedyIDs.push_back(i);  // bitvektor setzen => ids und ~ids? // TODO: Powerfunktion auf -1 setzen und verwenden? Powerfunktion >= 0 ?

    updateKernelVector(basisFunction, inputMesh, v, x);

    double sqrtP = std::sqrt(pMax);  

    for(int j = 0; j < _inSize; ++j) { // Vektorisierung kaputt: Sehr langsam
      if(!centerBits.at(j)) {
        v(j) -= (_basisMatrix.block(j, 0, 1, n) *  _basisMatrix.block(i, 0, 1, n).transpose())(0,0);
        v(j) /= sqrtP;                                
        _powerFunction(j) -= v(j) * v(j);
      }
    }
    
    centerBits.at(i) = true;
    _basisMatrix.col(n) = v;

    _cut.block(n, 0, 1, n).noalias() = -_basisMatrix.block(i, 0, 1, n) * _cut.block(0, 0, n, n).triangularView<Eigen::Lower>();
    _cut(n,n) = 1;
    _cut.block(n, 0, 1, n+1) /= v(i);

    std::cout << "iteration = " << n << "\r";
  }

  //mesh::Mesh centerMesh("greedy-centers", inputMesh.getDimensions(), mesh::Mesh::MESH_ID_UNDEFINED);
  //centerMesh.vertices() = _centers;
  //io::ExportVTU exporter{"PGreedy", "exports", centerMesh, io::Export::ExportKind::TimeWindows, 1, /*Rank*/ 0, /*size*/ 1};
  //exporter.doExport(0, 0.0);

  _kernel_eval = buildEvaluationMatrix(basisFunction, outputMesh, inputMesh, _greedyIDs);
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd PGreedySolver<RADIAL_BASIS_FUNCTION_T>::solveConservative(const Eigen::VectorXd &inputData, Polynomial polynomial) const
{
  // Not implemented
  PRECICE_ASSERT(false);
  return Eigen::VectorXd();
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd PGreedySolver<RADIAL_BASIS_FUNCTION_T>::solveConsistent(Eigen::VectorXd &inputData, Polynomial polynomial) const
{
  // First, compute the c vector
  // see https://eigen.tuxfamily.org/dox/group__TutorialSlicingIndexing.html

  size_t n = _greedyIDs.size();
  Eigen::VectorXd y = inputData(_greedyIDs);

  Eigen::VectorXd coeff      = _cut.block(0, 0, n, n).triangularView<Eigen::Lower>().transpose() * (_cut.block(0, 0, n, n).triangularView<Eigen::Lower>() * y);
  Eigen::VectorXd prediction = _kernel_eval.transpose() * coeff;

  return prediction;
}


template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedySolver<RADIAL_BASIS_FUNCTION_T>::clear()
{
  _centers.clear();
  _greedyIDs.clear();
  _cut         = Eigen::MatrixXd();
  _kernel_eval = Eigen::MatrixXd();
  _inSize      = 0;
  _outSize     = 0;
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::Index PGreedySolver<RADIAL_BASIS_FUNCTION_T>::getInputSize() const
{
  return _inSize;
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::Index PGreedySolver<RADIAL_BASIS_FUNCTION_T>::getOutputSize() const
{
  return _outSize;
}
} // namespace mapping
} // namespace precice

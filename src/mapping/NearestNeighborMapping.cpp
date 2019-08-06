#include "NearestNeighborMapping.hpp"
#include "query/FindClosestVertex.hpp"
#include "utils/Helpers.hpp"
#include "mesh/RTree.hpp"
#include <Eigen/Core>
#include <boost/function_output_iterator.hpp>
#include "utils/Event.hpp"



namespace precice {
extern bool syncMode;

namespace mapping {

NearestNeighborMapping:: NearestNeighborMapping
(
  Constraint constraint,
  int        dimensions)
:
  Mapping(constraint, dimensions)
{
  setInputRequirement(Mapping::MeshRequirement::VERTEX);
  setOutputRequirement(Mapping::MeshRequirement::VERTEX);
}

void NearestNeighborMapping:: computeMapping()
{
  P_TRACE(input()->vertices().size());

  P_ASSERT(input().get() != nullptr);
  P_ASSERT(output().get() != nullptr);

  const std::string baseEvent = "map.nn.computeMapping.From" + input()->getName() + "To" + output()->getName();
  precice::utils::Event e(baseEvent, precice::syncMode);
  
  if (getConstraint() == CONSISTENT){
    P_DEBUG("Compute consistent mapping");
    precice::utils::Event e2(baseEvent+".getIndexOnVertices", precice::syncMode);
    auto rtree = mesh::rtree::getVertexRTree(input());
    e2.stop();
    size_t verticesSize = output()->vertices().size();
    _vertexIndices.resize(verticesSize);
    const mesh::Mesh::VertexContainer& outputVertices = output()->vertices();
    for ( size_t i=0; i < verticesSize; i++ ) {
        const Eigen::VectorXd& coords = outputVertices[i].getCoords();
        // Search for the output vertex inside the input mesh and add index to _vertexIndices
        rtree->query(boost::geometry::index::nearest(coords, 1),
                     boost::make_function_output_iterator([&](size_t const& val) {
                         _vertexIndices[i] =  input()->vertices()[val].getID();
                       }));
    }
  }
  else {
    P_ASSERT(getConstraint() == CONSERVATIVE, getConstraint());
    P_DEBUG("Compute conservative mapping");
    precice::utils::Event e2(baseEvent+".getIndexOnVertices", precice::syncMode);
    auto rtree = mesh::rtree::getVertexRTree(output());
    e2.stop();
    size_t verticesSize = input()->vertices().size();
    _vertexIndices.resize(verticesSize);
    const mesh::Mesh::VertexContainer& inputVertices = input()->vertices();
    for ( size_t i=0; i < verticesSize; i++ ){
      const Eigen::VectorXd& coords = inputVertices[i].getCoords();
      // Search for the input vertex inside the output mesh and add index to _vertexIndices
      rtree->query(boost::geometry::index::nearest(coords, 1),
                   boost::make_function_output_iterator([&](size_t const& val) {
                       _vertexIndices[i] =  output()->vertices()[val].getID();
                     }));
    }
  }
  _hasComputedMapping = true;
}

bool NearestNeighborMapping:: hasComputedMapping() const
{
  P_TRACE(_hasComputedMapping);
  return _hasComputedMapping;
}

void NearestNeighborMapping:: clear()
{
  P_TRACE();
  _vertexIndices.clear();
  _hasComputedMapping = false;
  if (getConstraint() == CONSISTENT){
    mesh::rtree::clear(*input()); 
  } else {
    mesh::rtree::clear(*output()); 
  }
}

void NearestNeighborMapping:: map
(
  int inputDataID,
  int outputDataID )
{
  P_TRACE(inputDataID, outputDataID);

  precice::utils::Event e("map.nn.mapData.From" + input()->getName() + "To" + output()->getName(), precice::syncMode);

  const Eigen::VectorXd& inputValues = input()->data(inputDataID)->values();
  Eigen::VectorXd& outputValues = output()->data(outputDataID)->values();
  //assign(outputValues) = 0.0;
  int valueDimensions = input()->data(inputDataID)->getDimensions();
  P_ASSERT( valueDimensions == output()->data(outputDataID)->getDimensions(),
              valueDimensions, output()->data(outputDataID)->getDimensions() );
  P_ASSERT( inputValues.size() / valueDimensions == (int)input()->vertices().size(),
               inputValues.size(), valueDimensions, input()->vertices().size() );
  P_ASSERT( outputValues.size() / valueDimensions == (int)output()->vertices().size(),
               outputValues.size(), valueDimensions, output()->vertices().size() );
  if (getConstraint() == CONSISTENT){
    P_DEBUG("Map consistent");
    size_t const outSize = output()->vertices().size();
    for ( size_t i=0; i < outSize; i++ ){
      int inputIndex = _vertexIndices[i] * valueDimensions;
      for ( int dim=0; dim < valueDimensions; dim++ ){
        outputValues((i*valueDimensions)+dim) = inputValues(inputIndex+dim);
      }
    }
  }
  else {
    P_ASSERT(getConstraint() == CONSERVATIVE, getConstraint());
    P_DEBUG("Map conservative");
    size_t const inSize = input()->vertices().size();
    for ( size_t i=0; i < inSize; i++ ){
      int const outputIndex = _vertexIndices[i] * valueDimensions;
      for ( int dim=0; dim < valueDimensions; dim++ ){
        outputValues(outputIndex+dim) += inputValues((i*valueDimensions)+dim);
      }
    }
  }
}

void NearestNeighborMapping::tagMeshFirstRound()
{
  P_TRACE();
  precice::utils::Event e("map.nn.tagMeshFirstRound.From" + input()->getName() + "To" + output()->getName(), precice::syncMode);

  computeMapping();

  if (getConstraint() == CONSISTENT){
    for(mesh::Vertex& v : input()->vertices()){
      if(utils::contained(v.getID(),_vertexIndices)) v.tag();
    }
  }
  else {
    P_ASSERT(getConstraint() == CONSERVATIVE, getConstraint());
    for(mesh::Vertex& v : output()->vertices()){
      if(utils::contained(v.getID(),_vertexIndices)) v.tag();
    }
  }

  clear();
}

void NearestNeighborMapping::tagMeshSecondRound()
{
  P_TRACE();
  // for NN mapping no operation needed here
}

}} // namespace precice, mapping

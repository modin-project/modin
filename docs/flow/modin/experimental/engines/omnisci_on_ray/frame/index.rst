:orphan:

OmniSciOnRay Frame Implementation
=================================

Modin implements ``Frame``, ``PartitionManager``, ``AxisPartition`` and ``Partition`` classes
specific for ``OmniSciOnRay`` backend:

* :doc:`Frame <data>`
* :doc:`Partition <partition>`
* :doc:`AxisPartition <axis_partition>`
* :doc:`PartitionManager <partition_manager>`

Overview of OmniSci embedded engine usage can be accessed in the related section:

* :doc:`OmniSci Engine </flow/modin/experimental/engines/omnisci_on_ray/omnisci_engine>`

To support lazy execution Modin uses two types of trees. Operations on frames are described
by ``DFAlgNode`` based trees. Scalar computations are described by ``BaseExpr`` based tree.

* :doc:`DFAlgNode <df_algebra>`
* :doc:`BaseExpr <expr>`

Interactions with OmniSci engine are done using ``OmnisciServer`` class. Queries use serialized
Calcite relational algebra format. Calcite algebra nodes are based on ``CalciteBaseNode`` class.
Translation is done by ``CalciteBuilder`` class. Serialization is performed by ``CalciteSerializer``
class.

* :doc:`CalciteBaseNode <calcite_algebra>`
* :doc:`CalciteBuilder <calcite_builder>`
* :doc:`CalciteSerializer <calcite_serializer>`
* :doc:`OmnisciServer <omnisci_worker>`

.. toctree::
    :hidden:

    data
    partition
    axis_partition
    partition_manager
    ../omnisci_engine
    df_algebra
    expr
    calcite_algebra
    calcite_builder
    calcite_serializer
    omnisci_worker

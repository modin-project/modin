Why Modin?
==========

In this section, we explain the design and motivation behind Modin and why you should use Modin to scale up your pandas workflows. We first describe the architectural differences between pandas and Modin. Then we describe how Modin can also help resolve out-of-memory issues common to pandas. Finally, we look at the key differences between Modin and other distributed dataframe libraries. 

.. toctree::
    :maxdepth: 4
    
    pandas
    out_of_core
    modin_vs_dask_vs_koalas

Modin is built on many years of research and development at UC Berkeley. For more information on how this works underneath the hoods, check out our publications in this space:

- `Flexible Rule-Based Decomposition and Metadata Independence in Modin <https://people.eecs.berkeley.edu/~totemtang/paper/Modin.pdf>`_ (VLDB 2021)
- `Enhancing the Interactivity of Dataframe Queries by Leveraging Think Time <https://arxiv.org/pdf/2103.02145.pdf>`_ (IEEE Data Eng 2021)
- `Dataframe Systems: Theory, Architecture, and Implementation <https://www2.eecs.berkeley.edu/Pubs/TechRpts/2021/EECS-2021-193.pdf>`_ (PhD Dissertation 2021)
- `Scaling Data Science does not mean Scaling Machines <http://cidrdb.org/cidr2021/papers/cidr2021_abstract11.pdf>`_ (CIDR 2021)
- `Towards Scalable Dataframe Systems <https://arxiv.org/pdf/2001.00888.pdf>`_ (VLDB 2020)

"""Experimental graph alternatives; not drop-in replacements for the work chains.

Requires the pinned aiida-core fork in ``.github/workflows/graph.yml``. See
``reports/graph-migration.md`` for feature coverage, limitations, and verification.
The legacy workflow entry points remain unchanged.
"""

from aiida_quantumespresso.workflows.graphs.base import (
    bands_base_graph,
    matdyn_base_graph,
    neb_base_graph,
    ph_base_graph,
    pw_base_graph,
    q2r_base_graph,
)
from aiida_quantumespresso.workflows.graphs.cleanup import cleanup_graph_workdirs
from aiida_quantumespresso.workflows.graphs.ph import get_ph_recover_chain, merge_ph_graph_outputs
from aiida_quantumespresso.workflows.graphs.protocols import get_inputs_from_protocol
from aiida_quantumespresso.workflows.graphs.pdos import (
    pdos_from_parent,
    pdos_from_parent_parallel,
    pdos_from_parent_serial,
    pdos_from_scratch,
    pdos_from_scratch_parallel,
    pdos_from_scratch_serial,
)
from aiida_quantumespresso.workflows.graphs.pw import pw_bands_graph, pw_relax_graph

__all__ = (
    'bands_base_graph',
    'cleanup_graph_workdirs',
    'get_inputs_from_protocol',
    'get_ph_recover_chain',
    'matdyn_base_graph',
    'merge_ph_graph_outputs',
    'neb_base_graph',
    'pdos_from_parent',
    'pdos_from_parent_parallel',
    'pdos_from_parent_serial',
    'pdos_from_scratch',
    'pdos_from_scratch_parallel',
    'pdos_from_scratch_serial',
    'ph_base_graph',
    'pw_bands_graph',
    'pw_base_graph',
    'pw_relax_graph',
    'q2r_base_graph',
)

# Native QE graphs: current status

QE graph CI pins [`origin/graph-demo-vc-v4` at `db31187a5`](https://github.com/agoscinski/aiida-core/commit/db31187a536da235c2359286717658b378f30ff3).
The native graphs use source-order execution by default, postfix placement aliases, `partial` for shared PW retry inputs, declared `Err` returns for terminal restart failures, and `forward_outputs` for PDOS output namespaces. Legacy WorkChain entry points and scheduling are unchanged.

## Resolved on the v4 pin

### Parallel PDOS post-processing

Core's `dataflow()` (`cfc7aa1c2`) lets the independent DOS and PROJWFC calculations in [`pdos_stage_parallel`](../src/aiida_quantumespresso/workflows/graphs/pdos.py) overlap once the NSCF run passes validation and the energy windows are resolved. Cleanup and reporting wait for both to succeed. If one fails, the graph waits for started siblings without running the success continuation. The serial stages still sweep the DOS remote before starting PROJWFC. Placement names and `runtime_config` addresses did not change.

Unlike the original proposed `parallel()` syntax, `dataflow()` also allows dependencies within the block: a dependent placement waits for its producer. DOS and PROJWFC have no such dependency on each other.

### Preview a prepared launch

Core's `build(prepared_run=...)` (`db31187a5`) lets QE callers inspect a protocol launch without starting any calculation:

```python
prepared = prepare_run_from_protocol(
    'pw_bands', structure=structure, scf_code=code, bands_code=code, pseudos=pseudos
)
preview = pw_bands_graph.build(prepared_run=prepared)
print(preview)
```

The preview shows placement addresses, resolved inputs such as codes and scheduler options, and deferred inputs such as the bands parent folder. Branch arms and loop bodies are shown as static regions, without predicting the selected branch or number of attempts. This is not a CalcJob dry run. `build()` without inputs still only validates the definition.

## Still open

- **PW monitors:** Production still uses separate monitored and unmonitored placements. A focused test shows that a single placement can omit a branch-selected monitor; consolidation would also need to account for both existing runtime-config addresses.
- **Composite exit codes and output parity:** PW, PH, and NEB return declared statuses for terminal retry failures. PW 501/710 outcomes still expose usable outputs to callers that accept them. Composite stage-specific exit codes and exact legacy PDOS output parity remain open.
- **PH merge provenance:** Exact legacy merge provenance remains open.
- **Process cancellation:** Cleanup already refuses to act on active descendants. The [child-cancellation issue](../../reports/process-kill-skips-children.md) is a separate process-engine bug.

From Stdlib Require Import List.
From RocqSched Require Import Operational.Common.Step.
From RocqSched Require Import Operational.Awkernel.Minimal.CapturedTraceSyntax.
Import ListNotations.

Definition awk_generated_handoff_rows : list AwkernelSchedTraceEntry :=
  [ mkAwkernelSchedTraceEntry 0 (EvWakeup 1) None [1] false None
  ; mkAwkernelSchedTraceEntry 1 (EvRequestResched 1) None [1] true None
  ; mkAwkernelSchedTraceEntry 1 (EvHandleResched 1) None [1] true None
  ; mkAwkernelSchedTraceEntry 1 (EvChoose 1 1) None [1] true (Some 1)
  ; mkAwkernelSchedTraceEntry 1 (EvDispatch 1 1) (Some 1) [] false None
  ; mkAwkernelSchedTraceEntry 1 (EvComplete 1) None [] true None
  ].

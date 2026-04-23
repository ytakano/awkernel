From Stdlib Require Import List.
From RocqSched Require Import Operational.Common.Step.
From RocqSched Require Import Operational.Awkernel.Minimal.CapturedTraceSyntax.
Import ListNotations.

Definition awk_generated_handoff_rows : list AwkernelCapturedRow :=
  [ mkAwkernelCapturedRow 0 (EvWakeup 1) None [1] false None
  ; mkAwkernelCapturedRow 1 (EvChoose 1 1) None [1] false (Some 1)
  ; mkAwkernelCapturedRow 1 (EvDispatch 1 1) (Some 1) [] false None
  ; mkAwkernelCapturedRow 0 (EvWakeup 2) None [2] false None
  ; mkAwkernelCapturedRow 1 (EvChoose 1 2) None [2] false (Some 2)
  ; mkAwkernelCapturedRow 1 (EvDispatch 1 2) (Some 2) [2] false None
  ; mkAwkernelCapturedRow 0 (EvWakeup 2) None [2] false None
  ; mkAwkernelCapturedRow 1 (EvChoose 1 1) None [1; 2] false (Some 1)
  ; mkAwkernelCapturedRow 1 (EvDispatch 1 1) (Some 1) [2] false None
  ; mkAwkernelCapturedRow 0 (EvWakeup 3) None [2; 3] false None
  ; mkAwkernelCapturedRow 1 (EvChoose 1 3) None [2; 3] false (Some 3)
  ; mkAwkernelCapturedRow 1 (EvDispatch 1 3) (Some 3) [2; 3] false None
  ; mkAwkernelCapturedRow 0 (EvWakeup 3) None [2; 3] false None
  ; mkAwkernelCapturedRow 1 (EvChoose 1 2) None [2; 3] false (Some 2)
  ; mkAwkernelCapturedRow 1 (EvDispatch 1 2) (Some 2) [3] false None
  ; mkAwkernelCapturedRow 1 (EvComplete 2) None [3] true None
  ; mkAwkernelCapturedRow 1 (EvChoose 1 1) None [1; 3] false (Some 1)
  ; mkAwkernelCapturedRow 1 (EvDispatch 1 1) (Some 1) [3] false None
  ; mkAwkernelCapturedRow 0 (EvWakeup 4) None [3; 4] false None
  ; mkAwkernelCapturedRow 1 (EvChoose 1 4) None [3; 4] false (Some 4)
  ; mkAwkernelCapturedRow 1 (EvDispatch 1 4) (Some 4) [3; 4] false None
  ; mkAwkernelCapturedRow 0 (EvWakeup 4) None [3; 4] false None
  ; mkAwkernelCapturedRow 1 (EvChoose 1 3) None [3; 4] false (Some 3)
  ; mkAwkernelCapturedRow 1 (EvDispatch 1 3) (Some 3) [4] false None
  ; mkAwkernelCapturedRow 0 (EvWakeup 3) None [3; 4] false None
  ; mkAwkernelCapturedRow 1 (EvChoose 1 1) None [1; 3; 4] false (Some 1)
  ; mkAwkernelCapturedRow 1 (EvDispatch 1 1) (Some 1) [3; 4] false None
  ; mkAwkernelCapturedRow 1 (EvChoose 1 4) None [3; 4] false (Some 4)
  ; mkAwkernelCapturedRow 1 (EvDispatch 1 4) (Some 4) [3] false None
  ; mkAwkernelCapturedRow 1 (EvComplete 4) None [3] true None
  ; mkAwkernelCapturedRow 1 (EvChoose 1 3) None [3] false (Some 3)
  ; mkAwkernelCapturedRow 1 (EvDispatch 1 3) (Some 3) [] false None
  ; mkAwkernelCapturedRow 0 (EvWakeup 1) None [1] false None
  ; mkAwkernelCapturedRow 1 (EvComplete 3) None [1] true None
  ; mkAwkernelCapturedRow 1 (EvChoose 1 1) None [1] false (Some 1)
  ; mkAwkernelCapturedRow 1 (EvDispatch 1 1) (Some 1) [] false None
  ; mkAwkernelCapturedRow 1 (EvComplete 1) None [] true None
  ].

From Stdlib Require Import List Arith.PeanoNat.
From RocqSched Require Import Foundation.Base.
Import ListNotations.

Definition awk_generated_candidate_table : list (list JobId) :=
  [ [1]
  ; [1]
  ; [1]
  ; [2]
  ; [2]
  ; [2]
  ; [2]
  ; [1; 2]
  ; [1; 2]
  ; [2]
  ; [2]
  ; [1]
  ; [1]
  ; [1]
  ; [1]
  ; []
  ].

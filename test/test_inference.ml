(** Tests for Inference Engine *)

open Core
open Hardcaml
open Hardcaml_demo_project

module Simulator = Cyclesim.With_interface (Inference.Inference_engine.I) (Inference.Inference_engine.O)

let%expect_test "inference engine - beam hits splitter" =
  let scope = Scope.create ~flatten_design:true () in
  let sim = Simulator.create (Inference.Inference_engine.create scope) in
  let inputs = Cyclesim.inputs sim in
  let outputs = Cyclesim.outputs sim in

  (* Reset *)
  inputs.clear := Bits.vdd;
  Cyclesim.cycle sim;
  inputs.clear := Bits.gnd;

  (* Set up inputs: beam at position 8, splitter at position 8 *)
  inputs.beam_in := Bits.of_int_trunc ~width:16 0x0100;  (* Bit 8 set *)
  inputs.grid_in := Bits.of_int_trunc ~width:16 0x0100;  (* Splitter at pos 8 *)

  (* Start inference *)
  inputs.start := Bits.vdd;
  Cyclesim.cycle sim;
  inputs.start := Bits.gnd;

  (* Run until done - need 16 cycles for processing + 1 for output *)
  for _ = 1 to 20 do
    Cyclesim.cycle sim;
  done;

  (* Beam at 8 + splitter at 8 -> splits to positions 7 and 9 *)
  printf "Beam out: 0x%04x\n" (Bits.to_int_trunc !(outputs.beam_out));
  printf "Valid: %d\n" (Bits.to_int_trunc !(outputs.valid));
  printf "Split count: %d\n" (Bits.to_int_trunc !(outputs.split_count));

  (* Expected: beam_out = 0x0280 (bits 7 and 9 set from split) *)
  [%expect {|
    Beam out: 0x0280
    Valid: 0
    Split count: 1
    |}]

let%expect_test "inference engine - beam passes through" =
  let scope = Scope.create ~flatten_design:true () in
  let sim = Simulator.create (Inference.Inference_engine.create scope) in
  let inputs = Cyclesim.inputs sim in
  let outputs = Cyclesim.outputs sim in

  (* Reset *)
  inputs.clear := Bits.vdd;
  Cyclesim.cycle sim;
  inputs.clear := Bits.gnd;

  (* Beam at position 4, no splitter *)
  inputs.beam_in := Bits.of_int_trunc ~width:16 0x0010;  (* Bit 4 set *)
  inputs.grid_in := Bits.of_int_trunc ~width:16 0x0000;  (* No splitters *)

  inputs.start := Bits.vdd;
  Cyclesim.cycle sim;
  inputs.start := Bits.gnd;

  for _ = 1 to 20 do
    Cyclesim.cycle sim;
  done;

  printf "Beam out: 0x%04x\n" (Bits.to_int_trunc !(outputs.beam_out));
  printf "Split count: %d\n" (Bits.to_int_trunc !(outputs.split_count));

  (* Beam should pass through unchanged *)
  [%expect {|
    Beam out: 0x0010
    Split count: 0
  |}]

let%expect_test "mac unit" =
  let module Mac_sim = Cyclesim.With_interface (Inference.Mac.I) (Inference.Mac.O) in
  let scope = Scope.create () in
  let sim = Mac_sim.create (Inference.Mac.create scope) in
  let inputs = Cyclesim.inputs sim in
  let outputs = Cyclesim.outputs sim in

  (* Reset *)
  inputs.clear := Bits.vdd;
  Cyclesim.cycle sim;
  inputs.clear := Bits.gnd;

  (* Test: 10 * 5 + 0 = 50 *)
  inputs.enable := Bits.vdd;
  inputs.weight := Bits.of_int_trunc ~width:8 10;
  inputs.input := Bits.of_int_trunc ~width:16 5;
  inputs.acc_in := Bits.of_int_trunc ~width:32 0;
  Cyclesim.cycle sim;

  printf "MAC result: %d\n" (Bits.to_int_trunc !(outputs.acc_out));
  [%expect {| MAC result: 50 |}]

let%expect_test "relu activation" =
  let pos = Inference.Relu.apply (Signal.of_int_trunc ~width:16 100) in
  let neg = Inference.Relu.apply (Signal.of_int_trunc ~width:16 (-100)) in

  printf "ReLU(100) = %d\n" (Signal.to_int_trunc pos);
  printf "ReLU(-100) = %d\n" (Signal.to_int_trunc neg);
  [%expect {|
    ReLU(100) = 100
    ReLU(-100) = 0
  |}]

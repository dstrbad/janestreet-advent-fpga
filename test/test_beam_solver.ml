(** Tests for Beam Solver *)

open Core
open Hardcaml_demo_project

(* ============================================================ *)
(* Test Grids                                                   *)
(* ============================================================ *)

(* Simple grid: single splitter - S must be above ^ for beam to hit it *)
let simple_grid = {|
..S..
.....
..^..
.....
.....
|}

(* Diamond pattern: cascading splits *)
let diamond_grid = {|
....S....
.........
....^....
...^.^...
..^...^..
.^.....^.
|}

(* Edge case: immediate split *)
let immediate_split = {|
S
^
|}

(* Multiple levels *)
let multi_level = {|
..S..
.....
..^..
.^.^.
^...^
|}

(* ============================================================ *)
(* Tests                                                        *)
(* ============================================================ *)

let%expect_test "parse simple grid" =
  let grid = Beam_solver.parse_grid simple_grid in
  printf "Width: %d, Height: %d\n" grid.width grid.height;
  printf "Start: row=%d, col=%d\n" grid.start_row grid.start_col;
  [%expect {|
    Width: 5, Height: 5
    Start: row=0, col=2
  |}]

let%expect_test "simple grid simulation" =
  let grid = Beam_solver.parse_grid simple_grid in
  let result = Beam_solver.simulate grid in
  printf "Total splits: %d\n" result.total_splits;
  printf "Training samples: %d\n" (List.length result.training_data);
  [%expect {|
    Total splits: 1
    Training samples: 4
  |}]

let%expect_test "diamond grid simulation" =
  let grid = Beam_solver.parse_grid diamond_grid in
  let result = Beam_solver.simulate grid in
  printf "Total splits: %d\n" result.total_splits;
  printf "Training samples: %d\n" (List.length result.training_data);
  (* Diamond pattern: 1 + 2 + 2 + 2 = 7 splits expected *)
  [%expect {|
    Total splits: 7
    Training samples: 5
  |}]

let%expect_test "immediate split" =
  let grid = Beam_solver.parse_grid immediate_split in
  let result = Beam_solver.simulate grid in
  printf "Total splits: %d\n" result.total_splits;
  [%expect {|
    Total splits: 1
  |}]

let%expect_test "multi level simulation" =
  let grid = Beam_solver.parse_grid multi_level in
  let result = Beam_solver.simulate grid in
  printf "Total splits: %d\n" result.total_splits;

  (* Print training data for visualization *)
  List.iteri result.training_data ~f:(fun i sample ->
    let beam_str =
      Array.to_list sample.beam_state
      |> List.map ~f:(fun b -> if b then "1" else "0")
      |> String.concat
    in
    let next_str =
      Array.to_list sample.next_state
      |> List.map ~f:(fun b -> if b then "1" else "0")
      |> String.concat
    in
    printf "Row %d: beams=%s -> next=%s\n" i beam_str next_str);
  [%expect {|
    Total splits: 5
    Row 0: beams=00100 -> next=00100
    Row 1: beams=00100 -> next=01010
    Row 2: beams=01010 -> next=10101
    Row 3: beams=10101 -> next=01010
    |}]

let%expect_test "training data export" =
  let grid = Beam_solver.parse_grid immediate_split in
  let result = Beam_solver.simulate grid in
  let csv = Beam_solver.export_training_data result.training_data ~width:grid.width in
  print_endline csv;
  [%expect {|
    beam_0,grid_0,next_0
    1,1,0
  |}]

let%expect_test "visualization" =
  let grid = Beam_solver.parse_grid simple_grid in
  let beam_state = Array.create ~len:grid.width false in
  beam_state.(2) <- true;  (* Beam at column 2 *)
  let viz = Beam_solver.visualize grid beam_state ~row:2 in
  print_string viz;
  [%expect {|
    ..S..
    .....
    ..*..
    .....
    .....
    |}]

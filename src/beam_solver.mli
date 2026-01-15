(** Beam Solver for AoC 2025 Day 7: Tachyon Beam Splitters

    This module implements an algorithmic solver for the beam propagation problem.
    It also generates training data for the ML model.
*)

(** Grid cell types *)
type cell =
  | Empty    (** '.' - beam passes through *)
  | Splitter (** '^' - splits beam left and right *)
  | Start    (** 'S' - beam entry point *)

(** A parsed grid *)
type grid = {
  cells : cell array array;  (** 2D array of cells [row][col] *)
  width : int;
  height : int;
  start_col : int;           (** Column where beam enters *)
  start_row : int;           (** Row where beam enters *)
}

(** Training sample: input state -> output state *)
type training_sample = {
  beam_state : bool array;   (** Current beam positions *)
  grid_row : bool array;     (** Splitter positions in current row *)
  next_state : bool array;   (** Beam positions in next row *)
}

(** Result of simulation *)
type result = {
  total_splits : int;
  training_data : training_sample list;
}

(** Parse a grid from string input *)
val parse_grid : string -> grid

(** Simulate beam propagation through the grid.
    Returns total split count and training data. *)
val simulate : grid -> result

(** Export training data to a format suitable for PyTorch.
    Returns CSV-like string with input,output pairs. *)
val export_training_data : training_sample list -> width:int -> string

(** Run solver on input string, return split count *)
val solve : string -> int

(** Pretty-print the grid with beam state overlay *)
val visualize : grid -> bool array -> row:int -> string

(** Beam Solver for AoC 2025 Day 7: Tachyon Beam Splitters

    Algorithm:
    1. Find start position S
    2. Beam enters moving DOWN
    3. When beam hits '^', it splits into LEFT and RIGHT beams
    4. Left/Right beams also move DOWN each step
    5. Track beam positions row by row for training data
*)

open Core

(* ============================================================ *)
(* Types                                                        *)
(* ============================================================ *)

type cell =
  | Empty
  | Splitter
  | Start

type grid = {
  cells : cell array array;
  width : int;
  height : int;
  start_col : int;
  start_row : int;
}

type training_sample = {
  beam_state : bool array;
  grid_row : bool array;
  next_state : bool array;
}

type result = {
  total_splits : int;
  training_data : training_sample list;
}

(* Direction a beam is traveling horizontally *)
type h_direction =
  | GoingLeft
  | GoingRight
  | GoingDown  (* Straight down, no horizontal movement *)

(* A beam has a column position and horizontal direction *)
type beam = {
  col : int;
  h_dir : h_direction;
}

(* ============================================================ *)
(* Parsing                                                      *)
(* ============================================================ *)

let char_to_cell = function
  | '.' -> Empty
  | '^' -> Splitter
  | 'S' -> Start
  | c -> failwith (sprintf "Unknown cell character: %c" c)

let parse_grid (input : string) : grid =
  let lines =
    String.split_lines input
    |> List.filter ~f:(fun s -> not (String.is_empty (String.strip s)))
  in
  let height = List.length lines in
  if height = 0 then failwith "Empty grid";

  let width = String.length (List.hd_exn lines) in
  let cells = Array.make_matrix ~dimx:height ~dimy:width Empty in

  let start_col = ref (-1) in
  let start_row = ref (-1) in

  List.iteri lines ~f:(fun row line ->
    String.iteri line ~f:(fun col c ->
      let cell = char_to_cell c in
      cells.(row).(col) <- cell;
      if Poly.(cell = Start) then begin
        start_row := row;
        start_col := col
      end));

  if !start_col < 0 then failwith "No start position 'S' found in grid";

  { cells; width; height; start_col = !start_col; start_row = !start_row }

(* ============================================================ *)
(* Simulation                                                   *)
(* ============================================================ *)

(** Convert list of beams to boolean array of positions *)
let beams_to_state (beams : beam list) ~(width : int) : bool array =
  let state = Array.create ~len:width false in
  List.iter beams ~f:(fun b ->
    if b.col >= 0 && b.col < width then
      state.(b.col) <- true);
  state

(** Get splitter positions as boolean array *)
let row_to_splitters (grid : grid) ~(row : int) : bool array =
  Array.map grid.cells.(row) ~f:(fun cell ->
    match cell with
    | Splitter -> true
    | _ -> false)

(** Advance a beam one step:
    - If on a splitter: split into left and right
    - Otherwise: continue in current direction (and always move down)

    Note: The "down" movement is implicit - we process row by row.
    The horizontal movement happens within the same row before going down.
*)
let advance_beam (beam : beam) ~(is_splitter : bool) ~(width : int) : beam list =
  if is_splitter then
    (* Split into left and right *)
    let left_col = beam.col - 1 in
    let right_col = beam.col + 1 in
    let result = [] in
    let result =
      if left_col >= 0 then { col = left_col; h_dir = GoingLeft } :: result
      else result
    in
    let result =
      if right_col < width then { col = right_col; h_dir = GoingRight } :: result
      else result
    in
    result
  else
    (* Continue in current direction *)
    match beam.h_dir with
    | GoingDown ->
      (* Stay in same column *)
      [beam]
    | GoingLeft ->
      let new_col = beam.col - 1 in
      if new_col >= 0 then [{ beam with col = new_col }]
      else []  (* Beam exits grid *)
    | GoingRight ->
      let new_col = beam.col + 1 in
      if new_col < width then [{ beam with col = new_col }]
      else []  (* Beam exits grid *)

(** Process all beams for current row, return new beams and split count *)
let process_row (beams : beam list) ~(grid : grid) ~(row : int) : beam list * int =
  let width = grid.width in
  let splits = ref 0 in

  let new_beams =
    List.concat_map beams ~f:(fun beam ->
      let is_splitter =
        beam.col >= 0 && beam.col < width &&
        Poly.(grid.cells.(row).(beam.col) = Splitter)
      in
      if is_splitter then incr splits;
      advance_beam beam ~is_splitter ~width)
  in

  (* Deduplicate beams at same position *)
  let seen = Hash_set.create (module Int) in
  let unique_beams =
    List.filter new_beams ~f:(fun b ->
      if Hash_set.mem seen b.col then false
      else begin
        Hash_set.add seen b.col;
        true
      end)
  in

  (unique_beams, !splits)

(** Main simulation loop *)
let simulate (grid : grid) : result =
  let training_data = ref [] in
  let total_splits = ref 0 in

  (* Start with a single beam at the start position, going down *)
  let initial_beams = [{ col = grid.start_col; h_dir = GoingDown }] in

  let rec loop beams row =
    if row >= grid.height || List.is_empty beams then
      ()
    else begin
      (* Record current state *)
      let current_state = beams_to_state beams ~width:grid.width in
      let grid_row = row_to_splitters grid ~row in

      (* Process this row *)
      let new_beams, splits = process_row beams ~grid ~row in
      total_splits := !total_splits + splits;

      (* Record next state (for training) *)
      let next_state = beams_to_state new_beams ~width:grid.width in

      (* Store training sample *)
      training_data := {
        beam_state = current_state;
        grid_row;
        next_state;
      } :: !training_data;

      (* Continue to next row *)
      loop new_beams (row + 1)
    end
  in

  (* Start from the row after the start position *)
  loop initial_beams (grid.start_row + 1);

  {
    total_splits = !total_splits;
    training_data = List.rev !training_data;
  }

(* ============================================================ *)
(* Training Data Export                                         *)
(* ============================================================ *)

let bool_array_to_csv (arr : bool array) : string =
  Array.to_list arr
  |> List.map ~f:(fun b -> if b then "1" else "0")
  |> String.concat ~sep:","

let export_training_data (samples : training_sample list) ~(width : int) : string =
  let header =
    let beam_cols = List.init width ~f:(fun i -> sprintf "beam_%d" i) in
    let grid_cols = List.init width ~f:(fun i -> sprintf "grid_%d" i) in
    let next_cols = List.init width ~f:(fun i -> sprintf "next_%d" i) in
    String.concat ~sep:"," (beam_cols @ grid_cols @ next_cols)
  in
  let rows =
    List.map samples ~f:(fun s ->
      let beam_str = bool_array_to_csv s.beam_state in
      let grid_str = bool_array_to_csv s.grid_row in
      let next_str = bool_array_to_csv s.next_state in
      sprintf "%s,%s,%s" beam_str grid_str next_str)
  in
  String.concat ~sep:"\n" (header :: rows)

(* ============================================================ *)
(* Convenience Functions                                        *)
(* ============================================================ *)

let solve (input : string) : int =
  let grid = parse_grid input in
  let result = simulate grid in
  result.total_splits

let visualize (grid : grid) (beam_state : bool array) ~(row : int) : string =
  let buf = Buffer.create 256 in

  for r = 0 to grid.height - 1 do
    for c = 0 to grid.width - 1 do
      let cell_char =
        match grid.cells.(r).(c) with
        | Empty -> '.'
        | Splitter -> '^'
        | Start -> 'S'
      in
      let display =
        if r = row && beam_state.(c) then '*'
        else cell_char
      in
      Buffer.add_char buf display
    done;
    Buffer.add_char buf '\n'
  done;

  Buffer.contents buf

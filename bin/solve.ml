open! Core
open! Hardcaml_demo_project

(** Run the beam solver on a grid file and export training data *)

let solve_command =
  Command.basic
    ~summary:"Solve a beam grid and print the split count"
    [%map_open.Command
      let input_file = anon ("INPUT_FILE" %: string) in
      fun () ->
        let input = In_channel.read_all input_file in
        let result = Beam_solver.solve input in
        printf "Total splits: %d\n" result]
;;

let export_command =
  Command.basic
    ~summary:"Export training data from a grid file"
    [%map_open.Command
      let input_file = anon ("INPUT_FILE" %: string)
      and output_file = flag "-o" (optional string) ~doc:"FILE Output CSV file (default: stdout)"
      in
      fun () ->
        let input = In_channel.read_all input_file in
        let grid = Beam_solver.parse_grid input in
        let result = Beam_solver.simulate grid in
        let csv = Beam_solver.export_training_data result.training_data ~width:grid.width in
        match output_file with
        | Some file -> Out_channel.write_all file ~data:csv
        | None -> print_endline csv]
;;

let visualize_command =
  Command.basic
    ~summary:"Visualize beam propagation step by step"
    [%map_open.Command
      let input_file = anon ("INPUT_FILE" %: string) in
      fun () ->
        let input = In_channel.read_all input_file in
        let grid = Beam_solver.parse_grid input in
        let result = Beam_solver.simulate grid in

        printf "Grid: %d x %d\n" grid.width grid.height;
        printf "Start: row=%d, col=%d\n\n" grid.start_row grid.start_col;

        List.iteri result.training_data ~f:(fun i sample ->
          printf "=== Step %d ===\n" (i + 1);
          let viz = Beam_solver.visualize grid sample.beam_state ~row:(grid.start_row + 1 + i) in
          print_string viz;
          printf "\n");

        printf "Total splits: %d\n" result.total_splits]
;;

let generate_synthetic_command =
  Command.basic
    ~summary:"Generate synthetic training data from random grids"
    [%map_open.Command
      let width = flag "-w" (optional_with_default 16 int) ~doc:"INT Grid width (default: 16)"
      and height = flag "-h" (optional_with_default 16 int) ~doc:"INT Grid height (default: 16)"
      and count = flag "-n" (optional_with_default 100 int) ~doc:"INT Number of grids (default: 100)"
      and density = flag "-d" (optional_with_default 0.1 float) ~doc:"FLOAT Splitter density (default: 0.1)"
      and output_file = flag "-o" (optional string) ~doc:"FILE Output CSV file (default: stdout)"
      in
      fun () ->
        let all_samples = ref [] in

        for _ = 1 to count do
          (* Generate random grid *)
          let cells = Array.init height ~f:(fun row ->
            Array.init width ~f:(fun col ->
              if row = 0 && col = width / 2 then Beam_solver.Start
              else if row > 0 && Float.(<) (Random.float 1.0) density then Beam_solver.Splitter
              else Beam_solver.Empty))
          in
          let grid = {
            Beam_solver.cells;
            width;
            height;
            start_col = width / 2;
            start_row = 0;
          } in
          let result = Beam_solver.simulate grid in
          all_samples := result.training_data @ !all_samples
        done;

        let csv = Beam_solver.export_training_data !all_samples ~width in
        match output_file with
        | Some file ->
          Out_channel.write_all file ~data:csv;
          printf "Wrote %d samples to %s\n" (List.length !all_samples) file
        | None -> print_endline csv]
;;

let () =
  Command_unix.run
    (Command.group
       ~summary:"Beam solver for AoC 2025 Day 7"
       [ "solve", solve_command
       ; "export", export_command
       ; "visualize", visualize_command
       ; "generate", generate_synthetic_command
       ])
;;

open! Core
open! Hardcaml
open! Hardcaml_demo_project

let generate_range_finder_rtl () =
  let module C = Circuit.With_interface (Range_finder.I) (Range_finder.O) in
  let scope = Scope.create ~auto_label_hierarchical_ports:true () in
  let circuit = C.create_exn ~name:"range_finder_top" (Range_finder.hierarchical scope) in
  let rtl_circuits =
    Rtl.create ~database:(Scope.circuit_database scope) Verilog [ circuit ]
  in
  let rtl = Rtl.full_hierarchy rtl_circuits |> Rope.to_string in
  print_endline rtl
;;

let generate_inference_rtl () =
  let module C = Circuit.With_interface (Inference.Inference_engine.I) (Inference.Inference_engine.O) in
  let scope = Scope.create ~auto_label_hierarchical_ports:true () in
  let circuit = C.create_exn ~name:"beam_inference" (Inference.Inference_engine.hierarchical scope) in
  let rtl_circuits =
    Rtl.create ~database:(Scope.circuit_database scope) Verilog [ circuit ]
  in
  let rtl = Rtl.full_hierarchy rtl_circuits |> Rope.to_string in
  print_endline rtl
;;

let range_finder_rtl_command =
  Command.basic
    ~summary:"Generate Verilog for range finder example"
    [%map_open.Command
      let () = return () in
      fun () -> generate_range_finder_rtl ()]
;;

let inference_rtl_command =
  Command.basic
    ~summary:"Generate Verilog for beam inference engine"
    [%map_open.Command
      let () = return () in
      fun () -> generate_inference_rtl ()]
;;

let () =
  Command_unix.run
    (Command.group
      ~summary:"RTL generation for AoC Day 7"
      [ "range-finder", range_finder_rtl_command
      ; "inference", inference_rtl_command
      ])
;;

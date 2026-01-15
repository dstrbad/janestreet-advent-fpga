(** Hardcaml Inference Engine for Beam CNN

    Implements fixed-point Conv1D inference on FPGA.
    Architecture:
    - Input: 16 positions × 2 channels (beams + grid)
    - Conv1D(k=3, 2→8) + ReLU
    - Conv1D(k=3, 8→8) + ReLU
    - Conv1D(k=3, 8→1) + Sigmoid
    - Output: 16 positions × 1 (next beams)
*)

open Hardcaml

(** Fixed-point configuration *)
module Fixed : sig
  val data_bits : int      (** Bits for activations *)
  val weight_bits : int    (** Bits for weights (INT8) *)
  val accum_bits : int     (** Bits for accumulator *)
  val frac_bits : int      (** Fractional bits for fixed-point *)
end

(** MAC (Multiply-Accumulate) unit *)
module Mac : sig
  module I : sig
    type 'a t = {
      clock : 'a;
      clear : 'a;
      enable : 'a;
      weight : 'a;  (** INT8 weight *)
      input : 'a;   (** Fixed-point input *)
      acc_in : 'a;  (** Accumulator input *)
    } [@@deriving hardcaml]
  end

  module O : sig
    type 'a t = {
      acc_out : 'a; (** Accumulated result *)
    } [@@deriving hardcaml]
  end

  val create : Scope.t -> Signal.t I.t -> Signal.t O.t
end

(** Conv1D layer - processes one output position at a time *)
module Conv1d : sig
  module I : sig
    type 'a t = {
      clock : 'a;
      clear : 'a;
      start : 'a;           (** Start computation *)
      input_data : 'a;      (** Input channel data (serialized) *)
      weights : 'a;         (** Weight data (serialized) *)
      bias : 'a;            (** Bias value *)
    } [@@deriving hardcaml]
  end

  module O : sig
    type 'a t = {
      output : 'a;          (** Convolution output *)
      output_valid : 'a;    (** Output is valid *)
      done_ : 'a;           (** Layer computation complete *)
    } [@@deriving hardcaml]
  end

  val create :
    kernel_size:int ->
    in_channels:int ->
    Scope.t -> Signal.t I.t -> Signal.t O.t
end

(** ReLU activation: max(0, x) *)
module Relu : sig
  val apply : Signal.t -> Signal.t
end

(** Sigmoid approximation for fixed-point *)
module Sigmoid : sig
  val apply : Signal.t -> Signal.t
end

(** Top-level inference engine *)
module Inference_engine : sig
  module I : sig
    type 'a t = {
      clock : 'a;
      clear : 'a;
      start : 'a;           (** Start inference *)
      beam_in : 'a;         (** Input beam state (16 bits, 1 per position) *)
      grid_in : 'a;         (** Grid splitter state (16 bits) *)
    } [@@deriving hardcaml]
  end

  module O : sig
    type 'a t = {
      beam_out : 'a;        (** Output beam state (16 bits) *)
      valid : 'a;           (** Output is valid *)
      done_ : 'a;           (** Inference complete *)
      split_count : 'a;     (** Number of splits detected *)
    } [@@deriving hardcaml]
  end

  val create : Scope.t -> Signal.t I.t -> Signal.t O.t
  val hierarchical : Scope.t -> Signal.t I.t -> Signal.t O.t
end

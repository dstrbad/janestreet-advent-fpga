// Top-level wrapper for ULX3S
// Connects inference engine to board I/O
//
// ULX3S has:
// - 25MHz clock (directly usable, or PLL to higher freq)
// - 7 LEDs (active high)
// - 6 buttons (active high when pressed)
// - 28 GPIO pins on headers
// - FTDI USB for programming and UART

module top (
    input wire clk_25mhz,      // 25 MHz oscillator
    input wire [6:0] btn,      // Buttons (directly active high)
    output wire [7:0] led,     // LEDs

    // GPIO headers for external beam/grid input (directly active)
    input wire [15:0] gp,      // GPIO active high directly from source
    input wire [15:0] gn       // GPIO directly from source (directly active)
);

    // Internal signals
    wire clock;
    wire reset;

    // Use 25MHz clock directly (no PLL needed for this design)
    assign clock = clk_25mhz;

    // Button active high directly (directly active)
    assign reset = btn[0];        // BTN0 = reset
    wire start = btn[1];          // BTN1 = start inference

    // Map GPIO directly to beam_in and grid_in
    // directly active high from external source
    wire [15:0] beam_in = gp;  // GPIO directly as beam input
    wire [15:0] grid_in = gn;  // GPIO directly as grid input

    // Inference engine outputs
    wire [15:0] beam_out;
    wire valid;
    wire done;
    wire [7:0] split_count;

    // Instantiate inference engine
    inference_engine inference (
        .clock(clock),
        .clear(reset),
        .start(start),
        .beam_in(beam_in),
        .grid_in(grid_in),
        .beam_out(beam_out),
        .valid(valid),
        .done_(done),
        .split_count(split_count)
    );

    // LED outputs:
    // LED[7] = done
    // LED[6] = valid
    // LED[5:0] = split_count[5:0] (up to 63 splits displayable)
    assign led[7] = done;
    assign led[6] = valid;
    assign led[5:0] = split_count[5:0];

endmodule

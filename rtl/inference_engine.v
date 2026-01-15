module inference_engine (
    grid_in,
    beam_in,
    clear,
    clock,
    start,
    beam_out,
    valid,
    done_,
    split_count
);

    input [15:0] grid_in;
    input [15:0] beam_in;
    input clear;
    input clock;
    input start;
    output [15:0] beam_out;
    output valid;
    output done_;
    output [7:0] split_count;

    wire [7:0] _22;
    wire _90;
    wire [3:0] _89;
    wire [4:0] _91;
    wire _87;
    wire [4:0] _88;
    wire [4:0] _92;
    wire _83;
    wire [4:0] _84;
    wire _80;
    wire [4:0] _81;
    wire [4:0] _85;
    wire [4:0] _93;
    wire _75;
    wire [4:0] _76;
    wire _72;
    wire [4:0] _73;
    wire [4:0] _77;
    wire _68;
    wire [4:0] _69;
    wire _65;
    wire [4:0] _66;
    wire [4:0] _70;
    wire [4:0] _78;
    wire [4:0] _94;
    wire _59;
    wire [4:0] _60;
    wire _56;
    wire [4:0] _57;
    wire [4:0] _61;
    wire _52;
    wire [4:0] _53;
    wire _49;
    wire [4:0] _50;
    wire [4:0] _54;
    wire [4:0] _62;
    wire _44;
    wire [4:0] _45;
    wire _41;
    wire [4:0] _42;
    wire [4:0] _46;
    wire _37;
    wire [4:0] _38;
    wire _34;
    wire [4:0] _35;
    wire [4:0] _39;
    wire [4:0] _47;
    wire [4:0] _63;
    wire [4:0] _95;
    wire [2:0] _31;
    wire [7:0] _96;
    wire [7:0] _97;
    wire [7:0] _25;
    reg [7:0] _99;
    wire [7:0] _1;
    reg [7:0] _23;
    wire _101;
    wire _104;
    wire _103;
    reg _106;
    wire _4;
    reg _105;
    wire [15:0] _115;
    wire [14:0] _124;
    wire [15:0] _125;
    wire [15:0] _33;
    wire [14:0] _119;
    wire [15:0] _121;
    wire [15:0] _7;
    wire [15:0] _117;
    wire [15:0] _9;
    wire [15:0] _118;
    wire [15:0] _122;
    wire [15:0] _126;
    wire [15:0] _127;
    wire [1:0] _20;
    wire [1:0] _100;
    wire [7:0] _29;
    wire _11;
    wire _13;
    wire [7:0] _109;
    wire [7:0] _110;
    wire [7:0] _108;
    reg [7:0] _111;
    wire [7:0] _14;
    reg [7:0] _28;
    wire _30;
    wire [1:0] _113;
    wire [1:0] _98;
    wire _16;
    wire [1:0] _112;
    reg [1:0] _114;
    wire [1:0] _17;
    (* fsm_encoding="one_hot" *)
    reg [1:0] _21;
    reg [15:0] _128;
    wire [15:0] _18;
    reg [15:0] _116;
    assign _22 = 8'b00000000;
    assign _90 = _33[0:0];
    assign _89 = 4'b0000;
    assign _91 = { _89,
                   _90 };
    assign _87 = _33[1:1];
    assign _88 = { _89,
                   _87 };
    assign _92 = _88 + _91;
    assign _83 = _33[2:2];
    assign _84 = { _89,
                   _83 };
    assign _80 = _33[3:3];
    assign _81 = { _89,
                   _80 };
    assign _85 = _81 + _84;
    assign _93 = _85 + _92;
    assign _75 = _33[4:4];
    assign _76 = { _89,
                   _75 };
    assign _72 = _33[5:5];
    assign _73 = { _89,
                   _72 };
    assign _77 = _73 + _76;
    assign _68 = _33[6:6];
    assign _69 = { _89,
                   _68 };
    assign _65 = _33[7:7];
    assign _66 = { _89,
                   _65 };
    assign _70 = _66 + _69;
    assign _78 = _70 + _77;
    assign _94 = _78 + _93;
    assign _59 = _33[8:8];
    assign _60 = { _89,
                   _59 };
    assign _56 = _33[9:9];
    assign _57 = { _89,
                   _56 };
    assign _61 = _57 + _60;
    assign _52 = _33[10:10];
    assign _53 = { _89,
                   _52 };
    assign _49 = _33[11:11];
    assign _50 = { _89,
                   _49 };
    assign _54 = _50 + _53;
    assign _62 = _54 + _61;
    assign _44 = _33[12:12];
    assign _45 = { _89,
                   _44 };
    assign _41 = _33[13:13];
    assign _42 = { _89,
                   _41 };
    assign _46 = _42 + _45;
    assign _37 = _33[14:14];
    assign _38 = { _89,
                   _37 };
    assign _34 = _33[15:15];
    assign _35 = { _89,
                   _34 };
    assign _39 = _35 + _38;
    assign _47 = _39 + _46;
    assign _63 = _47 + _62;
    assign _95 = _63 + _94;
    assign _31 = 3'b000;
    assign _96 = { _31,
                   _95 };
    assign _97 = _30 ? _96 : _23;
    assign _25 = _16 ? _22 : _23;
    always @* begin
        case (_21)
        2'b00:
            _99 <= _25;
        2'b01:
            _99 <= _97;
        default:
            _99 <= _23;
        endcase
    end
    assign _1 = _99;
    always @(posedge _13) begin
        if (_11)
            _23 <= _22;
        else
            _23 <= _1;
    end
    assign _101 = _100 == _21;
    assign _104 = 1'b0;
    assign _103 = 1'b1;
    always @* begin
        case (_21)
        2'b00:
            _106 <= _104;
        2'b10:
            _106 <= _103;
        default:
            _106 <= _105;
        endcase
    end
    assign _4 = _106;
    always @(posedge _13) begin
        if (_11)
            _105 <= _104;
        else
            _105 <= _4;
    end
    assign _115 = 16'b0000000000000000;
    assign _124 = _33[15:1];
    assign _125 = { _104,
                    _124 };
    assign _33 = _9 & _7;
    assign _119 = _33[14:0];
    assign _121 = { _119,
                    _104 };
    assign _7 = grid_in;
    assign _117 = ~ _7;
    assign _9 = beam_in;
    assign _118 = _9 & _117;
    assign _122 = _118 | _121;
    assign _126 = _122 | _125;
    assign _127 = _30 ? _126 : _116;
    assign _20 = 2'b00;
    assign _100 = 2'b10;
    assign _29 = 8'b00010000;
    assign _11 = clear;
    assign _13 = clock;
    assign _109 = 8'b00000001;
    assign _110 = _28 + _109;
    assign _108 = _16 ? _22 : _28;
    always @* begin
        case (_21)
        2'b00:
            _111 <= _108;
        2'b01:
            _111 <= _110;
        default:
            _111 <= _28;
        endcase
    end
    assign _14 = _111;
    always @(posedge _13) begin
        if (_11)
            _28 <= _22;
        else
            _28 <= _14;
    end
    assign _30 = _28 == _29;
    assign _113 = _30 ? _100 : _21;
    assign _98 = 2'b01;
    assign _16 = start;
    assign _112 = _16 ? _98 : _21;
    always @* begin
        case (_21)
        2'b00:
            _114 <= _112;
        2'b01:
            _114 <= _113;
        2'b10:
            _114 <= _20;
        default:
            _114 <= _21;
        endcase
    end
    assign _17 = _114;
    always @(posedge _13) begin
        if (_11)
            _21 <= _20;
        else
            _21 <= _17;
    end
    always @* begin
        case (_21)
        2'b01:
            _128 <= _127;
        default:
            _128 <= _116;
        endcase
    end
    assign _18 = _128;
    always @(posedge _13) begin
        if (_11)
            _116 <= _115;
        else
            _116 <= _18;
    end
    assign beam_out = _116;
    assign valid = _105;
    assign done_ = _101;
    assign split_count = _23;

endmodule
module beam_inference (
    grid_in,
    beam_in,
    start,
    clear,
    clock,
    beam_out,
    valid,
    done_,
    split_count
);

    input [15:0] grid_in;
    input [15:0] beam_in;
    input start;
    input clear;
    input clock;
    output [15:0] beam_out;
    output valid;
    output done_;
    output [7:0] split_count;

    wire [7:0] _16;
    wire _17;
    wire _18;
    wire [15:0] _5;
    wire [15:0] _7;
    wire _9;
    wire _11;
    wire _13;
    wire [25:0] _15;
    wire [15:0] _19;
    assign _16 = _15[25:18];
    assign _17 = _15[17:17];
    assign _18 = _15[16:16];
    assign _5 = grid_in;
    assign _7 = beam_in;
    assign _9 = start;
    assign _11 = clear;
    assign _13 = clock;
    inference_engine
        inference_engine
        ( .clock(_13),
          .clear(_11),
          .start(_9),
          .beam_in(_7),
          .grid_in(_5),
          .beam_out(_15[15:0]),
          .valid(_15[16:16]),
          .done_(_15[17:17]),
          .split_count(_15[25:18]) );
    assign _19 = _15[15:0];
    assign beam_out = _19;
    assign valid = _18;
    assign done_ = _17;
    assign split_count = _16;

endmodule


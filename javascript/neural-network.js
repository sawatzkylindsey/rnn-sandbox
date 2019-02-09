
var MEMORY_CHIP_HEIGHT = 5;
var MEMORY_CHIP_WIDTH = 2;
var total_width = 1200;
var layer_height = 225;
var input_width = 100;
var x_margin = 25;
var y_margin = 50;
var height = 20;
var w = 30;
var h = layer_height / 3.0;
if ((h / MEMORY_CHIP_HEIGHT) != (w / MEMORY_CHIP_WIDTH)) {
    throw "chips aren't square (" + (w / MEMORY_CHIP_WIDTH) + ", " + (h / MEMORY_CHIP_HEIGHT) + ")";
}
var operand_height = (h * 2.0 / 5.0);
var operator_height = (h - (operand_height * 2));
var black = "black";
var dark_grey = "#c8c8c8";
var light_grey = "#e1e1e1";
var dark_red = "#e60000";
var light_red = "#ff1919";
var debug = window.location.hash.substring(1) == "debug";
var svg = null;
var sequence = [];

$(document).ready(function () {
    svg = d3.select('body').append('svg')
        .style('position', 'absolute')
        .style('top', 0)
        .style('left', 0)
        .style("width", total_width + (total_width / 5) - 10)
        .style('height', layer_height * 25);

    // arrow head definition
    svg.insert('defs', ':first-child')
        .append('marker')
        .attr('id', 'arrow')
        .attr('markerUnits', 'strokeWidth')
        .attr('markerWidth', 12)
        .attr('markerHeight', 12)
        .attr('viewBox', '0 0 12 12')
        .attr('refX', 12)
        .attr('refY', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M2,2 L10,6 L2,10 L6,6 L2,2')
        .style('fill', light_grey);

    d3.json("words")
        .get(function (error, data) { drawAutocomplete(0, data); });
});

function drawTimestep(fake_timestep, data) {
    console.log("Timestep (fake, actual): (" + fake_timestep + ", " + data.timestep + ")");
    console.log(data);
    var timestep = data.timestep;
    $(".timestep-" + timestep).remove();

    if (data.x_word != sequence[timestep]) {
        svg.append("text")
            .attr("class", "timestep-" + timestep)
            .attr("x", x_margin + (input_width * 2 / 3))
            .attr("y", y_margin + (timestep * layer_height) + h + height + 5)
            .style("font-size", "14px")
            .style("fill", black)
            .text(data.x_word);
    }

    var x_offset = (x_margin * 2) + input_width;
    var y_offset = y_margin + (timestep * layer_height);
    var operand_height = (h * 2.0 / 5.0);
    var operator_height = (h - (operand_height * 2));

    if (debug) {
        // gridlines
        for (var x = 0; x <= total_width; x += w) {
            svg.append("line")
                .attr("class", "timestep-" + timestep)
                .attr("x1", x_offset + x - 0.05)
                .attr("y1", y_offset)
                .attr("x2", x_offset + x - 0.05)
                .attr("y2", y_offset + layer_height)
                .attr("stroke-dasharray", "5,5")
                .attr("stroke", black)
                .attr("stroke-width", 0.1);
        }
        for (var y = 0; y <= layer_height; y += (h / 2.0)) {
            svg.append("line")
                .attr("class", "timestep-" + timestep)
                .attr("x1", x_offset)
                .attr("y1", y_offset + y - 0.05)
                .attr("x2", x_offset + total_width)
                .attr("y2", y_offset + y - 0.05)
                .attr("stroke-dasharray", "5,5")
                .attr("stroke", black)
                .attr("stroke-width", 0.1);
        }
    }

    //draw embedding
    drawHiddenState(getGeometry(timestep, "embedding"), data.embedding);

    // Draw units
    for (var u = 0; u < data.units.length; u++) {
        var unit_offset = u * w * 16;

        /*drawVline(timestep, x_offset + (w * 13) + (w / 2) + unit_offset, y_margin + (timestep * layer_height) + (h * 3 / 2),
            x_offset + (w * 4) + unit_offset, y_offset + layer_height);*/

        drawHiddenState(getGeometry(timestep, "cell_previous", u), data.units[u].cell_previous);
        /*drawMultiplication(timestep, x_offset + (w * 3) + (w) - (operator_height / 2) + unit_offset, y_offset + (h * 1 / 2) - (operator_height / 2), operator_height,
            [data.units[u].cell_previous, data.units[u].forget_gate, data.units[u].forget], u, null);
        var forget_gate = getGeometry(timestep, "forget_gate", u);
        drawHiddenState(forget_gate, data.units[u].forget_gate);
        drawGate(timestep, forget_gate.x - w + 6, forget_gate.y + (operand_height / 4), w * 2 / 4);
        drawHline(timestep, x_offset + (w * 4) + (operator_height / 2) + unit_offset, y_offset + (h / 2),
            x_offset + (w * 5) + unit_offset, y_offset + (h / 2));*/
        drawHiddenState(getGeometry(timestep, "forget", u), data.units[u].forget);
        /*drawHline(timestep, x_offset + (w * 2) + unit_offset, y_offset + h,
            x_offset + (w * 7) + (w / 2) + unit_offset, y_offset + h + (operand_height / 2),
            x_offset + (w * 2) + unit_offset + ((w * 3 / 2) / 2), y_offset + h + (operand_height / 4));*/

        /*drawVline(timestep, x_offset + (w * 17) + (w / 2) + unit_offset, y_margin + (timestep * layer_height) + (h * 3 / 2),
            x_offset + (w * 8) + unit_offset, y_offset + h + layer_height);*/

        drawHiddenState(getGeometry(timestep, "input_hat", u), data.units[u].input_hat);
        /*drawMultiplication(timestep, x_offset + (w * 7) + (w) - (operator_height / 2) + unit_offset, y_offset + (h * 3 / 2) - (operator_height / 2), operator_height,
            [data.units[u].input_hat, data.units[u].remember_gate, data.units[u].remember], u, null);
        var remember_gate = getGeometry(timestep, "remember_gate", u);
        drawHiddenState(remember_gate, data.units[u].remember_gate);
        drawGate(timestep, remember_gate.x - w + 6, remember_gate.y + (operand_height / 4), w * 2 / 4);
        drawHline(timestep, x_offset + (w * 8) + (operator_height / 2) + unit_offset, y_offset + (h * 3 / 2),
            x_offset + (w * 9) + unit_offset, y_offset + (h * 3 / 2));*/
        drawHiddenState(getGeometry(timestep, "remember", u), data.units[u].remember);
        /*drawHline(timestep, x_offset + (w * 6) + unit_offset, y_offset + (h / 2),
            x_offset + (w * 11) + (w / 2) + unit_offset, y_offset + (h / 2) + (operand_height / 2));
        drawHiddenState(getGeometry(timestep, "forget_hat", u), data.units[u].forget);
        drawHline(timestep, x_offset + (w * 10) + unit_offset, y_offset + (h * 3 / 2),
            x_offset + (w * 11) + (w / 2) + unit_offset, y_offset + h + (operand_height / 2) + (operator_height / 2));
        drawAddition(timestep, x_offset + (w * 11) + (w) - (operator_height / 2) + unit_offset, y_offset + (h) - (operator_height / 2), operator_height,
            [data.units[u].forget, data.units[u].remember, data.units[u].cell], u, null);
        drawHiddenState(getGeometry(timestep, "remember_hat", u), data.units[u].remember);
        drawHline(timestep, x_offset + (w * 12) + (operator_height / 2) + unit_offset, y_offset + (h * 2 / 2),
            x_offset + (w * 13) + unit_offset, y_offset + (h * 2 / 2));
        drawHiddenState(getGeometry(timestep, "cell", u), data.units[u].cell);
        drawHline(timestep, x_offset + (w * 14) + unit_offset, y_offset + h,
            x_offset + (w * 15) + (w / 2) + unit_offset, y_offset + (h / 2) + (operand_height / 2));*/
        drawHiddenState(getGeometry(timestep, "cell_hat", u), data.units[u].cell_hat);
        /*drawMultiplication(timestep, x_offset + (w * 15) + (w) - (operator_height / 2) + unit_offset, y_offset + (h) - (operator_height / 2), operator_height,
            [data.units[u].cell_hat, data.units[u].output_gate, data.units[u].output], u, null);
        var output_gate = getGeometry(timestep, "output_gate", u);
        drawHiddenState(output_gate, data.units[u].output_gate);
        drawGate(timestep, output_gate.x - w + 6, output_gate.y + (operand_height / 4), w * 2 / 4);
        drawHline(timestep, x_offset + (w * 16) + (operator_height / 2) + unit_offset, y_offset + (h * 2 / 2),
            x_offset + (w * 17) + unit_offset, y_offset + (h * 2 / 2));*/
        drawHiddenState(getGeometry(timestep, "output", u), data.units[u].output);
    }

    // Draw softmax
    /*drawHline(timestep, x_offset + (data.units.length * w * 17), y_offset + (h * 2 / 2),
        x_offset + (data.units.length * w * 17) + (w * 3 / 2), y_offset + (h * 2 / 2));*/
    drawSoftmax(getGeometry(timestep, "softmax"), data.softmax, 1.0);

    svg.append("rect")
        .attr("class", "timestep-" + timestep)
        .attr("x", x_offset + (2 * w * 18) + (w * 3 / 2) + x_margin)
        .attr("y", y_offset + h - (height / 2) - 1)
        .attr("width", input_width)
        .attr("height", height)
        .style("fill", "#dee0e2");
    svg.append("text")
        .attr("class", "timestep-" + timestep)
        .attr("x", x_offset + (2 * w * 18) + (w * 3 / 2) + x_margin + 2)
        .attr("y", y_offset + h + 5)
        .style("font-size", "17px")
        .style("fill", black)
        .style("background-color", "#dee0e2")
        .text(data.y_word);
}

function drawHiddenState(geometry, wv, class_suffix) {
    drawStateWidget(geometry, wv.minimum, wv.maximum, wv.vector, wv.colour, wv.predictions, class_suffix);
}

function drawStateWidget(geometry, min, max, vector, colour, predictions, class_suffix) {
    if (min >= max) {
        throw "min " + min + " cannot exceed max " + max;
    }

    if (min > 0) {
        throw "min " + min + " cannot be greater than 0";
    }

    if (max < 0) {
        throw "max " + max + " cannot be less than 0";
    }

    var found_min = d3.min(vector, function(d) { return d.value; });
    if (found_min < min) {
        throw "found value " + found_min + " exceeding min " + min;
    }

    var found_max = d3.max(vector, function(d) { return d.value; });
    if (found_max > max) {
        throw "found value " + found_max + " exceeding max " + max;
    }

    var stroke_width = 1;

    var macro_y = d3.scaleBand()
        .padding(0.2)
        .domain(Array.from(Array(MEMORY_CHIP_HEIGHT).keys()))
        .range([geometry.y + (stroke_width / 2.0), geometry.y + geometry.height - (stroke_width / 2.0)]);
    function y(position) {
        return macro_y(position % MEMORY_CHIP_HEIGHT);
    }

    var macro_x = d3.scaleBand()
        .padding(0.2)
        .domain(Array.from(Array(MEMORY_CHIP_WIDTH).keys()))
        .range([geometry.x + (stroke_width / 2.0), geometry.x + geometry.width - (stroke_width / 2.0)]);
    function x(position) {
        return macro_x(Math.floor(position / MEMORY_CHIP_HEIGHT));
    }

    var magnitude = d3.scaleLinear()
        .domain([min, max])
        .range([0, macro_x.bandwidth()]);

    if (debug) {
        svg.append("text")
            .attr("class", "timestep-" + geometry.timestep + " " + class_suffix)
            .attr("x", geometry.x)
            .attr("y", geometry.y - 2)
            .style("font-size", "12px")
            .style("fill", "red")
            .text(geometry.name);
    }

    if (predictions != null) {
        var predictionGeometry = Object.assign({}, geometry);
        predictionGeometry.x += w + (w / 4);
        predictionGeometry.y += (h * 1 / 4);
        predictionGeometry.width = (w / 2);
        predictionGeometry.height = (h / 2);
        drawSoftmax(predictionGeometry, predictions, 0.5);
    }

    // boundary box
    svg.append("rect")
        .attr("class", "timestep-" + geometry.timestep + " " + class_suffix)
        .attr("x", geometry.x + 0.5)
        .attr("y", geometry.y + 0.5)
        .attr("width", geometry.width - 1)
        .attr("height", geometry.height - 1)
        .attr("stroke", light_grey)
        .attr("stroke-width", 1)
        .attr("fill", "none");
    // Chip's colour & magnitude.
    svg.selectAll(".chip")
        .data(vector)
        .enter()
            .append("rect")
            .attr("class", "timestep-" + geometry.timestep + " " + class_suffix)
            .attr("x", function (d) {
                var base_x = x(d.position);

                if (d.value >= 0) {
                    return base_x;
                } else {
                    return base_x + (macro_x.bandwidth() - magnitude(d.value));
                }
            })
            .attr("y", function (d) { return y(d.position); })
            .attr("width", function (d) { return magnitude(d.value); })
            .attr("height", macro_y.bandwidth())
            .attr("stroke", "none")
            .attr("fill", function(d) {
                return colour == "none" ? light_grey : colour;
            });
    // Chip's scaling box.
    svg.selectAll(".chip")
        .data(vector)
        .enter()
            .append("rect")
            .attr("class", "timestep-" + geometry.timestep + " " + class_suffix)
            .attr("x", function (d) { return x(d.position); })
            .attr("y", function (d) { return y(d.position); })
            .attr("width", macro_x.bandwidth())
            .attr("height", macro_y.bandwidth())
            .attr("stroke", light_grey)
            .attr("stroke-width", stroke_width)
            .attr("fill", "none");
    // Chip's direction line.
    svg.selectAll(".chip")
        .data(vector)
        .enter()
            .append("line")
            .attr("class", "timestep-" + geometry.timestep + " " + class_suffix)
            .attr("x1", function(d) {
                var base_x = x(d.position);

                if (d.value >= 0) {
                    return base_x;
                } else {
                    return base_x + macro_x.bandwidth();
                }
            })
            .attr("y1", function(d) { return y(d.position) - 1; })
            .attr("x2", function(d) {
                var base_x = x(d.position);

                if (d.value >= 0) {
                    return base_x;
                } else {
                    return base_x + macro_x.bandwidth();
                }
            })
            .attr("y2", function(d) { return y(d.position) + macro_y.bandwidth() + 1; })
            .attr("stroke", black)
            .attr("stroke-width", stroke_width);
    /*if (class_suffix == null) {
        svg.selectAll(".bar")
            .data(vector)
            .enter()
                .append("rect")
                .attr("id", function(d) { return "hoverbar-" + geometry.timestep + "-" + geometry.name + "-" + d.position; })
                .attr("class", "timestep-" + geometry.timestep)
                .attr("x", geometry.x + 1.5)
                .attr("y", function(d) { return y(d.position) + 1; })
                .attr("width", geometry.width - 3)
                .attr("height", y.bandwidth() - 2)
                .attr("stroke", black)
                .attr("stroke-width", 1)
                .attr("fill", colour == "none" ? "white" : colour)
                .style("opacity", 0)
                .on("mouseover", function(d) {
                    if (geometry.name != "embedding" && (geometry.timestep > 0 || !geometry.name.startsWith("cell_previous"))) {
                        d3.select("#hoverbar-" + geometry.timestep + "-" + geometry.name + "-" + d.position)
                            .style("opacity", 0.5);
                    }
                })
                .on("mouseout", function(d) {
                    if (geometry.name != "embedding" && (geometry.timestep > 0 || !geometry.name.startsWith("cell_previous"))) {
                        d3.select("#hoverbar-" + geometry.timestep + "-" + geometry.name + "-" + d.position)
                            .style("opacity", 0);
                    }
                })
                .on("click", function(d) {
                    var source = {
                        x: geometry.x + 1,
                        y: y(d.position) + 1,
                        width: geometry.width - 2,
                        height: y.bandwidth() - 2,
                    }
                    if (geometry.name != "embedding" && (geometry.timestep > 0 || !geometry.name.startsWith("cell_previous"))) {
                        var slice = sequence.slice(0, geometry.timestep + 1);
                        console.log(geometry.name);
                        d3.json("weight-explain?" + slice.map(s => "sequence=" + encodeURI(s)).join("&") + "&name=" + geometry.name + "&column=" + d.column)
                            .get(function (error, we) {
                                drawExplain(geometry.timestep, source, we, colour);
                            });
                    }
                });
    }*/
}

function drawSoftmax(geometry, labelWeightVector, opacity) {
    var min = labelWeightVector.minimum;
    var max = labelWeightVector.maximum;
    var vector = labelWeightVector.vector;

    var found_min = d3.min(vector, function(d) { return d.value; });
    if (found_min < min) {
        throw "found value " + found_min + " exceeding min " + min;
    }

    var found_max = d3.max(vector, function(d) { return d.value; });
    if (found_max > max) {
        throw "found value " + found_max + " exceeding max " + max;
    }

    var stroke_width = 1;

    var y = d3.scaleBand()
        .domain(vector.map(function (d) { return d.position; }))
        .range([geometry.y + (stroke_width / 2.0), geometry.y + geometry.height - (stroke_width / 2.0)]);

    var x = d3.scaleLinear()
        .domain([min, max])
        .range([geometry.x + (stroke_width / 2.0), geometry.x + geometry.width - (stroke_width / 2.0)]);

    if (debug) {
        svg.append("text")
            .attr("class", "timestep-" + geometry.timestep)
            .attr("x", geometry.x)
            .attr("y", geometry.y - 2)
            .style("font-size", "12px")
            .style("fill", "red")
            .style("opacity", "opacity")
            .text(geometry.name);
    }

    // boundary box
    svg.append("rect")
        .attr("class", "timestep-" + geometry.timestep)
        .attr("x", geometry.x + 0.5)
        .attr("y", geometry.y + 0.5)
        .attr("width", geometry.width - 1)
        .attr("height", geometry.height - 1)
        .attr("stroke", light_grey)
        .attr("stroke-width", 1)
        .attr("fill", "none")
        .style("opacity", opacity);
    svg.append("line")
        .attr("class", "timestep-" + geometry.timestep)
        .attr("x1", x(0))
        .attr("y1", y.range()[0] - 2)   // Make the center line stand out slightly by pushing it beyond the rectangle.
        .attr("x2", x(0))
        .attr("y2", y.range()[1] + 2)   // Make the center line stand out slightly by pushing it beyond the rectangle.
        .attr("stroke", black)
        .attr("stroke-width", stroke_width)
        .style("opacity", opacity);
    svg.selectAll(".bar")
        .data(vector)
        .enter()
            .append("rect")
            .attr("class", "timestep-" + geometry.timestep)
            .attr("x", function (d) {
                return x(Math.min(0, d.value));
            })
            .attr("y", function (d) {
                return y(d.position);
            })
            .attr("width", function (d) {
                return Math.abs(x(Math.min(0, d.value)) - x(Math.max(0, d.value)));
            })
            .attr("height", y.bandwidth())
            .attr("stroke", black)
            .attr("stroke-width", stroke_width)
            .attr("fill", function(d) {
                if ("colour" in d) {
                    return d.colour;
                }

                return "none";
            })
            .style("opacity", opacity);
    svg.selectAll(".bar")
        .data(vector)
        .enter()
            .append("text")
            .attr("class", "timestep-" + geometry.timestep)
            .attr("x", function (d) {
                return geometry.x + Math.abs(x(d.value) - x(min)) + 5;
            })
            .attr("y", function (d) {
                return y(d.position) + (y.step() / 2) + 4;
            })
            .style("font-size", "12px")
            .style("opacity", opacity)
            .text(function (d) { return d.label; });
    /*svg.selectAll(".bar")
        .data(vector)
        .enter()
            .append("rect")
            .attr("id", function(d) { return "hoverbar-" + geometry.timestep + "-" + geometry.name + "-" + d.position; })
            .attr("class", "timestep-" + geometry.timestep)
            .attr("x", geometry.x + 1.5)
            .attr("y", function(d) { return y(d.position) + 1; })
            .attr("width", geometry.width - 3)
            .attr("height", y.bandwidth() - 2)
            .attr("stroke", black)
            .attr("stroke-width", 1)
            .style("opacity", 0)
            .on("mouseover", function(d) {
                if (geometry.name != "embedding" && (geometry.timestep > 0 || !geometry.name.startsWith("cell_previous"))) {
                    d3.select("#hoverbar-" + geometry.timestep + "-" + geometry.name + "-" + d.position)
                        .style("opacity", 0.5);
                }
            })
            .on("mouseout", function(d) {
                if (geometry.name != "embedding" && (geometry.timestep > 0 || !geometry.name.startsWith("cell_previous"))) {
                    d3.select("#hoverbar-" + geometry.timestep + "-" + geometry.name + "-" + d.position)
                        .style("opacity", 0);
                }
            })
            .on("click", function(d) {
                var source = {
                    x: geometry.x + 1,
                    y: y(d.position) + 1,
                    width: geometry.width - 2,
                    height: y.bandwidth() - 2,
                }
                if (geometry.name != "embedding" && (geometry.timestep > 0 || !geometry.name.startsWith("cell_previous"))) {
                    var slice = sequence.slice(0, geometry.timestep + 1);
                    console.log(geometry.name);
                    d3.json("weight-explain?" + slice.map(s => "sequence=" + encodeURI(s)).join("&") + "&name=" + geometry.name + "&column=" + d.column)
                        .get(function (error, we) {
                            drawExplain(geometry.timestep, source, we, null);
                        });
                }
            });*/
}

function drawExplain(timestep, source, we, colour) {
    console.log(we);
    $(".explain").remove();
    svg.append("rect")
        .attr("class", "timestep-" + timestep + " explain")
        .attr("x", source.x - 0.5)
        .attr("y", source.y - 1)
        .attr("width", source.width + 1)
        .attr("height", source.height + 2)
        .attr("stroke-width", 1)
        .attr("stroke", black)
        .attr("fill", colour)
        .style("opacity", 0.5);

    for (var key in we.vectors) {
        console.log(key);
        var parts = key.split("-");

        if (parts[0] == "output_previous") {
            drawExplainWidget(getGeometry(timestep - 1, "output", parts[1]), -we.bound, we.bound, we.vectors[key]);
        } else if (parts[0] == "cell_previous") {
            drawExplainWidget(getGeometry(timestep - 1, "cell", parts[1]), -we.bound, we.bound, we.vectors[key]);
        } else {
            drawExplainWidget(getGeometry(timestep, parts[0], parts[1]), -we.bound, we.bound, we.vectors[key]);
        }
    }
}

function drawExplainWidget(geometry, min, max, vector) {
    if (min > 0) {
        throw "min " + min + " cannot be greater than 0";
    }

    if (max < 0) {
        throw "max " + max + " cannot be less than 0";
    }

    var found_min = d3.min(vector, function(d) { return d.value; });
    if (found_min < min) {
        throw "found value " + found_min + " exceeding min " + min;
    }

    var found_max = d3.max(vector, function(d) { return d.value; });
    if (found_max > max) {
        throw "found value " + found_max + " exceeding max " + max;
    }

    var stroke_width = 1;
    var band_width = geometry.height / (vector.length * 3);

    var y = d3.scaleBand()
        .domain(vector.map(function (d) { return d.position; }))
        .range([geometry.y + (stroke_width / 2.0), geometry.y + geometry.height - (stroke_width / 2.0)]);

    var x = d3.scaleLinear()
        .domain([min, max])
        .range([geometry.x + (stroke_width / 2.0), geometry.x + geometry.width - (stroke_width / 2.0)]);

    // boundary box
    svg.append("rect")
        .attr("class", "timestep-" + geometry.timestep + " explain")
        .attr("x", geometry.x + 0.5)
        .attr("y", geometry.y + 0.5)
        .attr("width", geometry.width - 1)
        .attr("height", geometry.height - 1)
        .style("opacity", 0.5)
        .attr("stroke", black)
        .attr("stroke-width", 1)
        .attr("fill", "none");
    svg.append("line")
        .attr("class", "timestep-" + geometry.timestep + " explain")
        .attr("x1", x(0))
        .attr("y1", y.range()[0])
        .attr("x2", x(0))
        .attr("y2", y.range()[1])
        .style("opacity", 0.5)
        .attr("stroke", black)
        .attr("stroke-width", 1);
    svg.selectAll(".bar")
        .data(vector)
        .enter()
            .append("rect")
            .attr("class", "timestep-" + geometry.timestep + " explain")
            .attr("x", function (d) {
                return x(Math.min(0, d.value));
            })
            .attr("y", function (d) {
                return y(d.position) + ((y.bandwidth() - band_width) / 2);
            })
            .attr("width", function (d) {
                return Math.abs(x(Math.min(0, d.value)) - x(Math.max(0, d.value)));
            })
            .attr("height", band_width)
            .style("opacity", 0.5)
            .attr("stroke", black)
            .attr("stroke-width", stroke_width);
}

function drawHline(timestep, x1, y1, x2, y2) {
    drawHline(timestep, x1, y1, x2, y2, null);
}

function drawHline(timestep, x1, y1, x2, y2, x_midpoint) {
    var line_data = [{x: x1, y: y1}];
    var delta_x = Math.abs(x1 - x2);
    var delta_y = Math.abs(y1 - y2);
    var sharpness = 10;

    if (x_midpoint == null && delta_y != 0) {
        if ((delta_x / 2) < sharpness) {
            throw "bad line";
        }
        line_data.push({x: x1 + (delta_x / 2) - sharpness - 1, y: y1});
        line_data.push({x: x1 + (delta_x / 2) - sharpness, y: y1});
        line_data.push({x: x1 + (delta_x / 2), y: y1 + ((y2 > y1 ? 1 : -1) * (delta_y / 2))});
        line_data.push({x: x1 + (delta_x / 2) + sharpness, y: y2});
        line_data.push({x: x1 + (delta_x / 2) + sharpness + 1, y: y2});
    } else if (x_midpoint != null) {
        if ((x_midpoint - x1) < sharpness) {
            throw "bad line";
        }
        line_data.push({x: x_midpoint - sharpness - 1, y: y1});
        line_data.push({x: x_midpoint - sharpness, y: y1});
        line_data.push({x: x_midpoint, y: y1 + ((y2 > y1 ? 1 : -1) * (delta_y / 2))});
        line_data.push({x: x_midpoint + sharpness, y: y2});
        line_data.push({x: x_midpoint + sharpness + 1, y: y2});
    }

    line_data.push({x: x2, y: y2});

    if (debug) {
        for (var i = 0; i < line_data.length; i++) {
            svg.append("circle")
                .attr("class", "timestep-" + timestep)
                .attr("r", 2)
                .attr("cx", line_data[i]["x"])
                .attr("cy", line_data[i]["y"])
                .style("fill", "red");
        }
    }

    var pather = d3.line()
        .x(function(d) { return d["x"]; })
        .y(function(d) { return d["y"]; })
        .curve(d3.curveBasis);
        //.curve(d3.curveBundle.beta(.9));
    svg.selectAll(".bar")
        .data([line_data])
        .enter()
            .append("path")
            .attr("class", "timestep-" + timestep)
            .attr("d", pather)
            .attr("stroke", light_grey)
            .attr("stroke-width", 1)
            .attr("marker-end", "url(#arrow)")
            .style("fill", "none");
}

function drawVline(timestep, x1, y1, x2, y2) {
    drawVline(timestep, x1, y1, x2, y2, null);
}

function drawVline(timestep, x1, y1, x2, y2, y_midpoint) {
    var line_data = [{x: x1, y: y1}];
    var delta_x = Math.abs(x1 - x2);
    var delta_y = Math.abs(y1 - y2);
    var sharpness = 10;

    if (y_midpoint == null && delta_x != 0) {
        if ((delta_y / 2) < sharpness) {
            throw "bad line";
        }
        line_data.push({x: x1, y: y1 + (delta_y / 2) - sharpness - 1});
        line_data.push({x: x1, y: y1 + (delta_y / 2) - sharpness});
        line_data.push({x: x1 + ((x2 > x1 ? 1 : -1) * (delta_x / 2)), y: y1 + (delta_y / 2)});
        line_data.push({x: x2, y: y1 + (delta_y / 2) + sharpness});
        line_data.push({x: x2, y: y1 + (delta_y / 2) + sharpness + 1});
    } else if (y_midpoint != null) {
        if ((y_midpoint - y1) < sharpness) {
            throw "bad line";
        }
        line_data.push({x: x1, y: y_midpoint - sharpness - 1});
        line_data.push({x: x1, y: y_midpiont - sharpness});
        line_data.push({x: x1 + ((x2 > x1 ? 1 : -1) * (delta_x / 2)), y: y_midpoint});
        line_data.push({x: x2, y: y_midpoint + sharpness});
        line_data.push({x: x2, y: y_midpoint + sharpness + 1});
    }

    line_data.push({x: x2, y: y2});

    if (debug) {
        for (var i = 0; i < line_data.length; i++) {
            svg.append("circle")
                .attr("class", "timestep-" + timestep)
                .attr("r", 2)
                .attr("cx", line_data[i]["x"])
                .attr("cy", line_data[i]["y"])
                .style("fill", "red");
        }
    }

    var pather = d3.line()
        .x(function(d) { return d["x"]; })
        .y(function(d) { return d["y"]; })
        .curve(d3.curveBasis);
        //.curve(d3.curveBundle.beta(.9));
    svg.selectAll(".bar")
        .data([line_data])
        .enter()
            .append("path")
            .attr("class", "timestep-" + timestep)
            .attr("d", pather)
            .attr("stroke", light_grey)
            .attr("stroke-width", 1)
            .attr("marker-end", "url(#arrow)")
            .style("fill", "none");
}

function drawOperatorCircle(timestep, x_offset, y_offset, size, addition, parts, class_suffix) {
    svg.append("circle")
        .attr("class", "timestep-" + timestep + " " + class_suffix)
        .attr("cx", x_offset + (size / 2))
        .attr("cy", y_offset + (size / 2))
        .attr("r", (size / 2) - 0.5)
        .attr("stroke", black)
        .attr("stroke-width", 1)
        .attr("fill", light_grey)
        .on("mouseover", function(d) {
            if (parts != null) {
                d3.event.target.style.fill = dark_grey;
            }
        })
        .on("mouseout", function(d) {
            if (parts != null) {
                d3.event.target.style.fill = light_grey;
            }
        })
        .on("click", function(d) {
            if (parts != null) {
                drawZoom(timestep, x_offset + (size / 2), addition, parts);
            }
        });
}

function drawAddition(timestep, x_offset, y_offset, size, parts, class_suffix) {
    drawOperatorCircle(timestep, x_offset, y_offset, size, true, parts, class_suffix);
    var stroke_width = size / 10;
    var stroke_length = size / 2;
    svg.append("line")
        .attr("class", "timestep-" + timestep + " " + class_suffix)
        .attr("x1", x_offset + ((size - stroke_length) / 2))
        .attr("y1", y_offset + (size / 2))
        .attr("x2", x_offset + ((size - stroke_length) / 2) + stroke_length)
        .attr("y2", y_offset + (size / 2))
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
    svg.append("line")
        .attr("class", "timestep-" + timestep + " " + class_suffix)
        .attr("x1", x_offset + (size / 2))
        .attr("y1", y_offset + ((size - stroke_length) / 2))
        .attr("x2", x_offset + (size / 2))
        .attr("y2", y_offset + ((size - stroke_length) / 2) + stroke_length)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
}

function drawMultiplication(timestep, x_offset, y_offset, size, parts, class_suffix) {
    drawOperatorCircle(timestep, x_offset, y_offset, size, false, parts, class_suffix);
    var stroke_width = size / 10;
    var stroke_length = size / 2;
    var qq = Math.sqrt((stroke_length**2) / 2);
    svg.append("line")
        .attr("class", "timestep-" + timestep + " " + class_suffix)
        .attr("x1", x_offset + ((size - qq) / 2))
        .attr("y1", y_offset + ((size - qq) / 2))
        .attr("x2", x_offset + ((size - qq) / 2) + qq)
        .attr("y2", y_offset + ((size - qq) / 2) + qq)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
    svg.append("line")
        .attr("class", "timestep-" + timestep + " " + class_suffix)
        .attr("x1", x_offset + ((size - qq) / 2))
        .attr("y1", y_offset + ((size - qq) / 2) + qq)
        .attr("x2", x_offset + ((size - qq) / 2) + qq)
        .attr("y2", y_offset + ((size - qq) / 2))
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
}

function drawEquals(timestep, x_offset, y_offset, size, class_suffix) {
    drawOperatorCircle(timestep, x_offset, y_offset, size, false, null, class_suffix);
    var stroke_width = size / 10;
    var stroke_length = size / 2;
    svg.append("line")
        .attr("class", "timestep-" + timestep + " " + class_suffix)
        .attr("x1", x_offset + ((size - stroke_length) / 2))
        .attr("y1", y_offset + (size / 2) - (stroke_width * 1))
        .attr("x2", x_offset + ((size - stroke_length) / 2) + stroke_length)
        .attr("y2", y_offset + (size / 2) - (stroke_width * 1))
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
    svg.append("line")
        .attr("class", "timestep-" + timestep + " " + class_suffix)
        .attr("x1", x_offset + ((size - stroke_length) / 2))
        .attr("y1", y_offset + (size / 2) + (stroke_width * 1))
        .attr("x2", x_offset + ((size - stroke_length) / 2) + stroke_length)
        .attr("y2", y_offset + (size / 2) + (stroke_width * 1))
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
}

function drawGate(timestep, x_offset, y_offset, size) {
    var stroke_width = size/50;

    //left vertical bar
    svg.append("rect")
        .attr("class", "timestep-" + timestep)
        .attr("x", x_offset)
        .attr("y", y_offset+size/8)
        .attr("width", size/10)
        .attr("height", size-size/8)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width)
        .attr("fill", "#FFF");

    //left circle
    svg.append("circle")
        .attr("class", "timestep-" + timestep)
        .attr("cx", x_offset+size*2/39)
        .attr("cy", y_offset+size/18)
        .attr("r", size/16)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width)
        .attr("fill", "#FFF");

    //right vertical bar
    svg.append("rect")
        .attr("class", "timestep-" + timestep)
        .attr("x", x_offset+size*3/2-size/8)
        .attr("y", y_offset+size/8)
        .attr("width", size/10)
        .attr("height", size-size/8)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width)
        .attr("fill", "#FFF");

    //right circle
    svg.append("circle")
        .attr("class", "timestep-" + timestep)
        .attr("cx", x_offset+size*3/2-size*2/27)
        .attr("cy", y_offset+size/18)
        .attr("r", size/16)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width)
        .attr("fill", "#FFF"); 

    //left door
    svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset+size*13/20)
        .attr("y1", y_offset)
        .attr("x2", x_offset+size*13/20)
        .attr("y2", y_offset+size-size/6)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);

   svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset+size*13/20)
        .attr("y1", y_offset)
        .attr("x2", x_offset+size/10)
        .attr("y2", y_offset+size/6)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);

   svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset+size*13/20)
        .attr("y1", y_offset+size-size/6)
        .attr("x2", x_offset+size/10)
        .attr("y2", y_offset+size-size/8)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);

   //right door
   svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset+size*4/5)
        .attr("y1", y_offset)
        .attr("x2", x_offset+size*4/5)
        .attr("y2", y_offset+size-size/6)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);

   svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset+size*4/5)
        .attr("y1", y_offset)
        .attr("x2", x_offset+size*3/2-size/8)
        .attr("y2", y_offset+size/6)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);

   svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset+size*4/5)
        .attr("y1", y_offset+size-size/6)
        .attr("x2", x_offset+size*3/2-size/8)
        .attr("y2", y_offset+size-size/8)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
}

function drawZoom(timestep, x_middle, addition, parts) {
    var x_offset = x_middle - ((w * 4.5) / 2) - (operator_height * 2);
    var y_offset = y_margin + (timestep * layer_height);
    var zoom_class = "zoom-" + Math.random().toString(36).substring(5);
    svg.append("rect")
        .attr("class", "timestep-" + timestep + " " + zoom_class)
        .attr("x", x_offset)
        .attr("y", y_offset)
        .attr("width", (w * 4.5) + (operator_height * 4))
        .attr("height", h * 2)
        .attr("stroke", black)
        .attr("stroke-width", 1)
        .attr("fill", "white");
    var close_x_offset = x_offset + (w * 4.5) + (operator_height * 4);
    var close_y_offset = y_offset;
    svg.append("circle")
        .attr("class", "timestep-" + timestep + " " + zoom_class)
        .attr("cx", close_x_offset)
        .attr("cy", close_y_offset)
        .attr("r", (operator_height / 2) - 0.5)
        .attr("stroke", dark_red)
        .attr("stroke-width", 1)
        .attr("fill", light_red)
        .on("mouseover", function(d) {
            if (parts != null) {
                d3.event.target.style.fill = dark_red;
            }
        })
        .on("mouseout", function(d) {
            if (parts != null) {
                d3.event.target.style.fill = light_red;
            }
        })
        .on("click", function(d) {
            $("." + zoom_class).remove();
        });
    var stroke_width = operator_height / 10;
    var stroke_length = operator_height / 2;
    var qq = Math.sqrt((stroke_length**2) / 2);
    svg.append("line")
        .attr("class", "timestep-" + timestep + " " + zoom_class)
        .attr("x1", close_x_offset - ((operator_height - qq) / 2))
        .attr("y1", close_y_offset - ((operator_height - qq) / 2))
        .attr("x2", close_x_offset + qq)
        .attr("y2", close_y_offset + qq)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
    svg.append("line")
        .attr("class", "timestep-" + timestep + " " + zoom_class)
        .attr("x1", close_x_offset - ((operator_height - qq) / 2))
        .attr("y1", close_y_offset + qq)
        .attr("x2", close_x_offset + qq)
        .attr("y2", close_y_offset - ((operator_height - qq) / 2))
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
    var geometry = {width: w * 1.5, timestep: timestep, x: x_offset + operator_height - 2, y: y_offset + (h * 0.25), height: h * 1.5};
    drawHiddenState(geometry, parts[0], zoom_class);

    if (addition) {
        drawAddition(timestep, geometry.x + (w * 1.5) + 1, y_offset + (h) - (operator_height / 2), operator_height, null, zoom_class);
    } else {
        drawMultiplication(timestep, geometry.x + (w * 1.5) + 1, y_offset + (h) - (operator_height / 2), operator_height, null, zoom_class);
    }

    geometry.x += (w * 1.5) + operator_height + 2;
    drawHiddenState(geometry, parts[1], zoom_class);
    drawEquals(timestep, geometry.x + (w * 1.5) + 1, y_offset + (h) - (operator_height / 2), operator_height, zoom_class);
    geometry.x += (w * 1.5) + operator_height + 2;
    drawHiddenState(geometry, parts[2], zoom_class);
}

function drawAutocomplete(timestep, words) {
    var x_offset = x_margin;
    var y_offset = y_margin + (timestep * layer_height)
    var focus = null;

    svg.append("foreignObject")
        .attr("transform", "translate(" + x_offset + "," + (y_offset + h - (height / 2) - 1) + ")")
        .attr("width", input_width)
        .attr("height", height)
        .append("xhtml:div")
        .attr("id", "autocomplete-" + timestep);
    var autocomplete = $("#autocomplete-" + timestep);
    autocomplete.append("<input/>");
    autocomplete.on("input", function() {
        autocomplete.find(".autocomplete-option").remove();
        focus = -1;
        var value = autocomplete.find("input").val();

        if (value === "") {
            return false;
        }

        for (var i = 0; i < words.length; i++) {
            if (words[i].substr(0, value.length).toLowerCase() === value.toLowerCase()) {
                autocomplete.append("<div class='autocomplete-option'>" + words[i] + "</div>");
            }
        }

        $(".autocomplete-option").on("click", function(e) {
            autocomplete.find(".autocomplete-option").remove();
            var textContent = e.target.textContent.toLowerCase();
            autocomplete.find("input").val(textContent);

            if (timestep >= sequence.length) {
                sequence.push(textContent);
                d3.json("words")
                    .get(function (error, data) { drawAutocomplete(timestep + 1, data); });
            } else {
                sequence[timestep] = textContent;
            }

            drawWeightsFromSequence(timestep);
        });
    })
    .on("keydown", function(e) {
        var options = autocomplete.find(".autocomplete-option").length;

        // Down key
        if (e.keyCode === 40) {
            if (focus == options - 1) {
                focus = -1;
            } else {
                focus += 1;
            }
        }
        // Up key
        else if (e.keyCode === 38) {
            if (focus == -1) {
                focus = options - 1;
            } else {
                focus -= 1;
            }
        }
        // Enter key
        else if (e.keyCode == 13) {
            var selection = autocomplete.find(".autocomplete-active");

            if (selection.length == 1) {
                selection.click();
            } else {
                autocomplete.find(".autocomplete-option").remove();
                var textContent = autocomplete.find("input").val().toLowerCase();
                autocomplete.find("input").val(textContent);

                if (textContent == "") {
                    var tail = sequence.length;
                    sequence = sequence.slice(0, timestep);

                    for (var s = timestep; s <= tail; s++) {
                        if (s != timestep) {
                            $("#autocomplete-" + s).remove();
                        }

                        $(".timestep-" + s).remove();
                    }
                } else {
                    if (timestep >= sequence.length) {
                        sequence.push(textContent);
                        d3.json("words")
                            .get(function (error, data) { drawAutocomplete(timestep + 1, data); });
                    } else {
                        sequence[timestep] = textContent;
                    }

                    drawWeightsFromSequence(timestep);
                }
            }
        }

        autocomplete.find(".autocomplete-active").removeClass("autocomplete-active");

        if (focus >= 0) {
            autocomplete.find(".autocomplete-option:eq(" + focus + ")").addClass("autocomplete-active");
        }
    });
}

function drawWeightsFromSequence(timestep) {
    console.log("Full sequence: " + sequence);

    for (var s = timestep; s < sequence.length; s++) {
        var slice = sequence.slice(0, s + 1);
        console.log("Drawing sequence for " + (slice.length - 1) + ": " + slice);
        d3.json("weights?" + slice.map(s => "sequence=" + encodeURI(s)).join("&"))
            .get(function (error, data) { drawTimestep(slice.length - 1, data); });
    }
}

function getGeometry(timestep, name, layer) {
    var x_offset = (x_margin * 2) + input_width;
    var y_offset = y_margin + (timestep * layer_height);
    var layer_offset = layer * w * 14;
    var a = {width: w, timestep: timestep, name: name + (layer == null ? "" : "-" + layer)};
    var b;

    switch (name) {
        case "embedding":
            b = {x: x_offset + w, y: y_offset + (h * 3 / 4), height: h};
            break;
        case "cell_previous":
            b = {x: x_offset + (w * 9 / 2) + layer_offset, y: y_offset, height: h};
            break;
        /*case "forget_gate":
            b = {x: x_offset + (w * 3) + (w / 2) + layer_offset, y: y_offset + (h * 1 / 2) + (operator_height / 2), height: operand_height};
            break;*/
        case "forget":
            b = {x: x_offset + (w * 8) + layer_offset, y: y_offset, height: h};
            break;
        case "input_hat":
            b = {x: x_offset + (w * 9 / 2) + layer_offset, y: y_offset + (h * 3 / 2), height: h};
            break;
        /*case "remember_gate":
            b = {x: x_offset + (w * 7) + (w / 2) + layer_offset, y: y_offset + (h * 3 / 2) + (operator_height / 2), height: operand_height};
            break;*/
        case "remember":
            b = {x: x_offset + (w * 8) + layer_offset, y: y_offset + (h * 3 / 2), height: h};
            break;
        /*case "forget_hat":
            b = {x: x_offset + (w * 11) + (w / 2) + layer_offset, y: y_offset + (h * 1 / 2), height: operand_height};
            break;
        case "remember_hat":
            b = {x: x_offset + (w * 11) + (w / 2) + layer_offset, y: y_offset + (h * 2 / 2) + (operator_height / 2), height: operand_height};
            break;
        case "cell":
            b = {x: x_offset + (w * 13) + layer_offset, y: y_offset + (h * 1 / 2), height: h};
            break;*/
        case "cell_hat":
            b = {x: x_offset + (w * 11) + (w / 2) + layer_offset, y: y_offset + (h * 3 / 4), height: h};
            break;
        /*case "output_gate":
            b = {x: x_offset + (w * 15) + (w / 2) + layer_offset, y: y_offset + h + (operator_height / 2), height: operand_height};
            break;*/
        case "output":
            b = {x: x_offset + (w * 15) + layer_offset, y: y_offset + (h * 3 / 4), height: h};
            break;
        case "softmax":
            // For the 2 layers v
            b = {x: x_offset + (2 * w * 17) + (w * 3 / 2), y: y_offset + (h * 3 / 4), height: h};
            break;
    }

    return Object.assign({}, a, b);
}

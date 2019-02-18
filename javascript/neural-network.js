
var MEMORY_CHIP_HEIGHT = 5;
var MEMORY_CHIP_WIDTH = 2;
var total_width = 1400;
var total_height = 700;
var detail_margin = 10;
var layer_height = 225;
var input_width = 100;
var x_margin = 25;
var y_margin = 50;
var HEIGHT = 20;
var state_width = 30;
var state_height = layer_height / 3.0;
if ((state_height / MEMORY_CHIP_HEIGHT) != (state_width / MEMORY_CHIP_WIDTH)) {
    throw "chips aren't square (" + (state_width / MEMORY_CHIP_WIDTH) + ", " + (state_height / MEMORY_CHIP_HEIGHT) + ")";
}
var operand_height = (state_height * 2.0 / 5.0);
var operator_height = (state_height - (operand_height * 2));
var black = "#3f3f3f";
var dark_grey = "#7e7e7e";
var light_grey = "#bdbdbd";
var dark_red = "#e60000";
var light_red = "#ff1919";
var debug = window.location.hash.substring(1) == "debug";
var svg = null;
var main_sequence = [];
var main_timestep = null;
var compare_sequence = [];
var input_part = null;
var input_layer = null;

$(document).ready(function () {
    svg = d3.select('body').append('svg')
        .style('position', 'absolute')
        .style('top', 0)
        .style('left', 0)
        .style("width", total_width)
        .style('height', layer_height * 2);

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
    $("svg").height(layer_height * (main_sequence.length + 2));
    $(".timestep-" + data.timestep).remove();

    for (var t=0; t < main_sequence.length - 1; t++) {
        $(".timestep-" + t + ".softmax").remove();
    }

    if (data.x_word != main_sequence[data.timestep]) {
        svg.append("text")
            .attr("class", "timestep-" + data.timestep)
            .attr("x", x_margin + (input_width * 2 / 3))
            .attr("y", y_margin + (data.timestep * layer_height) + state_height + HEIGHT + 5)
            .style("font-size", "14px")
            .style("fill", black)
            .text(data.x_word);
    }


    var x_offset = (x_margin * 2) + input_width;
    var y_offset = y_margin + (data.timestep * layer_height);
    var operand_height = (state_height * 2.0 / 5.0);
    var operator_height = (state_height - (operand_height * 2));

    if (debug) {
        // gridlines
        for (var x = 0; x <= total_width; x += state_width) {
            svg.append("line")
                .attr("class", "timestep-" + data.timestep)
                .attr("x1", x_offset + x - 0.05)
                .attr("y1", y_offset)
                .attr("x2", x_offset + x - 0.05)
                .attr("y2", y_offset + layer_height)
                .attr("stroke-dasharray", "5,5")
                .attr("stroke", "blue")
                .attr("stroke-width", 0.1);
        }
        for (var y = 0; y <= layer_height; y += (state_height / 2.0)) {
            svg.append("line")
                .attr("class", "timestep-" + data.timestep)
                .attr("x1", x_offset)
                .attr("y1", y_offset + y - 0.05)
                .attr("x2", x_offset + total_width)
                .attr("y2", y_offset + y - 0.05)
                .attr("stroke-dasharray", "5,5")
                .attr("stroke", "blue")
                .attr("stroke-width", 0.1);
        }
    }

    // Draw embedding
    drawHiddenState(data, "embedding");

    // Draw units
    for (var part in data.units) {
        if (data.units.hasOwnProperty(part)) {
            for (var layer in data.units[part]) {
                if (data.units[part].hasOwnProperty(layer)) {
                    drawHiddenState(data, part, layer);
                }
            }
        }
    }

    // Draw softmax
    /*drawHline(timestep, x_offset + (data.units.length * w * 17), y_offset + (h * 2 / 2),
        x_offset + (data.units.length * w * 17) + (w * 3 / 2), y_offset + (h * 2 / 2));*/
    if (data.timestep == main_sequence.length - 1) {
        drawSoftmax(data, "softmax");
    }

    svg.append("rect")
        .attr("class", "timestep-" + data.timestep)
        .attr("x", x_offset + (2 * state_width * 18) + (state_width * 3 / 2) + x_margin)
        .attr("y", y_offset + state_height - (HEIGHT/ 2) - 1)
        .attr("width", input_width)
        .attr("height", HEIGHT)
        .style("fill", "#dee0e2");
    svg.append("text")
        .attr("class", "timestep-" + data.timestep)
        .attr("x", x_offset + (2 * state_width * 18) + (state_width * 3 / 2) + x_margin + 2)
        .attr("y", y_offset + state_height + 5)
        .style("font-size", "17px")
        .style("fill", black)
        .style("background-color", "#dee0e2")
        .text(data.y_word);
}

function drawHiddenState(data, part, layer) {
    var geometry = getGeometry(data.timestep, part, layer);

    if (geometry != null) {
        var hiddenState = null;

        if (part in data) {
            hiddenState = data[part];
        } else {
            hiddenState = data.units[part][layer];
        }

        var classes = "timestep-" + data.timestep;
        drawStateWidget(data.timestep, geometry, hiddenState.minimum, hiddenState.maximum, hiddenState.vector, hiddenState.colour, hiddenState.predictions, classes,
            MEMORY_CHIP_WIDTH, MEMORY_CHIP_HEIGHT, part, layer, null, null, null);
    }
}

function drawStateWidget(timestep, geometry, min, max, vector, colour, predictions, classes, chip_width, chip_height, part, layer, linker, linker_suffix, placement) {
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
        .domain(Array.from(Array(chip_height).keys()))
        .range([geometry.y + (stroke_width / 2.0), geometry.y + geometry.height - (stroke_width / 2.0)]);
    function y(position) {
        return macro_y(position % chip_height);
    }

    var macro_x = d3.scaleBand()
        .padding(0.2)
        .domain(Array.from(Array(chip_width).keys()))
        .range([geometry.x + (stroke_width / 2.0), geometry.x + geometry.width - (stroke_width / 2.0)]);
    function x(position) {
        return macro_x(Math.floor(position / chip_height));
    }

    var magnitude = d3.scaleLinear()
        .domain([min, max])
        .range([0, macro_x.bandwidth()]);

    if (debug) {
        svg.append("text")
            .attr("class", classes)
            .attr("x", geometry.x)
            .attr("y", geometry.y - 2)
            .style("font-size", "12px")
            .style("fill", "blue")
            .text("TODO");
    }

    var margin = (geometry.width / 6);

    if (predictions != null) {
        var predictionGeometry = {
            x: geometry.x + geometry.width + (geometry.width / 3) + margin,
            y: geometry.y + (geometry.height / 3),
            width: geometry.width / 3,
            height: geometry.height / 3,
        };
        drawPredictionWidget(timestep, predictionGeometry, predictions.minimum, predictions.maximum, predictions.vector, classes, true);
        // Draw colour prediction.
        svg.append("rect")
            .attr("class", classes)
            .attr("x", geometry.x + geometry.width + margin - (stroke_width / 2))
            .attr("y", geometry.y + (geometry.height * 1 / 4))
            .attr("width", (geometry.width / 3))
            .attr("height", (geometry.height / 2))
            .attr("stroke", "none")
            .attr("stroke-width", stroke_width)
            .attr("fill", colour == null ? "none" : colour)
            .style("opacity", 1.0);
    }

    if (timestep != null) {
        drawOpen(geometry.x + geometry.width + (margin * 2) - 0.5, geometry.y + margin, margin - 0.5, classes, function () {
            main_timestep = timestep;
            input_part = part;
            input_layer = layer;
            drawDetail();
        });
    }

    // boundary box
    svg.append("rect")
        .attr("class", classes)
        .attr("x", geometry.x + (stroke_width / 2.0))
        .attr("y", geometry.y + (stroke_width / 2.0))
        .attr("width", geometry.width - stroke_width)
        .attr("height", geometry.height - stroke_width)
        .attr("stroke", light_grey)
        .attr("stroke-width", stroke_width)
        .attr("fill", "none");
    // Chip's colour & magnitude.
    svg.selectAll(".chip")
        .data(vector)
        .enter()
            .append("rect")
            .attr("class", classes)
            .attr("x", function (d) {
                var base_x = x(d.position);

                if (d.value >= 0) {
                    return base_x;
                } else {
                    return base_x + (macro_x.bandwidth() - magnitude(d.value));
                }
            })
            .attr("y", function (d) {
                if (placement == null || placement == "top") {
                    return y(d.position);
                } else {
                    return y(d.position) + (macro_y.bandwidth() / 2) + 0.5;
                }
            })
            .attr("width", function (d) { return magnitude(d.value); })
            .attr("height", function (d) {
                if (placement == null) {
                    return macro_y.bandwidth();
                } else {
                    return (macro_y.bandwidth() / 2) - 1;
                }
            })
            .attr("stroke", "none")
            .attr("fill", dark_grey);
    // Chip's scaling box.
    svg.selectAll(".chip")
        .data(vector)
        .enter()
            .append("rect")
            .attr("class", function(d) { return classes + " linker-" + (linker == null ? d.position : linker[d.position]) + linker_suffix; })
            .attr("x", function (d) { return x(d.position); })
            .attr("y", function (d) {
                if (placement == null || placement == "top") {
                    return y(d.position);
                } else {
                    return y(d.position) + (macro_y.bandwidth() / 2) + 0.5;
                }
            })
            .attr("width", macro_x.bandwidth())
            .attr("height", function (d) {
                if (placement == null) {
                    return macro_y.bandwidth();
                } else {
                    return (macro_y.bandwidth() / 2) - 1;
                }
            })
            .attr("stroke", light_grey)
            .attr("stroke-width", stroke_width)
            .attr("fill", "transparent")
            .style("pointer-events", "bounding-box")
            .on("mouseover", function(d) {
                if (timestep == null) {
                    d3.selectAll(".linker-" + (linker == null ? d.position : linker[d.position]) + linker_suffix)
                        .transition()
                        .duration(100)
                        .attr("stroke", black)
                        .attr("stroke-width", stroke_width * 2);
                }
            })
            .on("mouseout", function(d) {
                if (timestep == null) {
                    d3.selectAll(".linker-" + (linker == null ? d.position : linker[d.position]) + linker_suffix)
                        .transition()
                        .duration(50)
                        .attr("stroke", light_grey)
                        .attr("stroke-width", stroke_width);
                }
            });
    // Chip's direction line.
    svg.selectAll(".chip")
        .data(vector)
        .enter()
            .append("line")
            .attr("class", classes)
            .attr("x1", function(d) {
                var base_x = x(d.position);

                if (d.value >= 0) {
                    return base_x;
                } else {
                    return base_x + macro_x.bandwidth();
                }
            })
            .attr("y1", function (d) {
                if (placement == null || placement == "top") {
                    return y(d.position) - 1;
                } else {
                    return y(d.position) + (macro_y.bandwidth() / 2) - 0.5;
                }
            })
            .attr("x2", function(d) {
                var base_x = x(d.position);

                if (d.value >= 0) {
                    return base_x;
                } else {
                    return base_x + macro_x.bandwidth();
                }
            })
            .attr("y2", function (d) {
                if (placement == null) {
                    return y(d.position) + macro_y.bandwidth() + 1;
                } else if (placement == "bottom") {
                    // Only necessary to manage rounding errors
                    return y(d.position) + macro_y.bandwidth() + 0.5;
                } else if (placement == "top") {
                    return y(d.position) + (macro_y.bandwidth() / 2);
                }
            })
            .attr("stroke", black)
            .attr("stroke-width", stroke_width);
}

function drawSoftmax(data, part) {
    var geometry = getGeometry(data.timestep, part, null);
    var labelWeightVector = data[part];
    var classes = "timestep-" + data.timestep;
    drawPredictionWidget(data.timestep, geometry, labelWeightVector.minimum, labelWeightVector.maximum, labelWeightVector.vector, classes, false);
}

function drawPredictionWidget(timestep, geometry, min, max, predictions, classes, subtle) {
    var found_min = d3.min(predictions, function(d) { return d.value; });
    if (found_min < min) {
        throw "found value " + found_min + " exceeding min " + min;
    }

    var found_max = d3.max(predictions, function(d) { return d.value; });
    if (found_max > max) {
        throw "found value " + found_max + " exceeding max " + max;
    }

    var stroke_width = 1;

    var y = d3.scaleBand()
        .domain(predictions.map(function (d) { return d.position; }))
        .range([geometry.y + (stroke_width / 2.0), geometry.y + geometry.height - (stroke_width / 2.0)]);

    var x = d3.scaleLinear()
        .domain([min, max])
        .range([geometry.x + (stroke_width / 2.0), geometry.x + geometry.width - (stroke_width / 2.0)]);

    var baseOpacity = subtle ? 0.2 : 1.0;
    var id_class = "softmax-" + Math.random().toString(36).substring(5);

    // boundary box
    svg.append("rect")
        .attr("class", classes + (subtle ? "" : " softmax") + " " + id_class)
        .attr("x", geometry.x + 0.5)
        .attr("y", geometry.y + 0.5)
        .attr("width", geometry.width - 1)
        .attr("height", geometry.height - 1)
        .attr("stroke", light_grey)
        .attr("stroke-width", 1)
        .attr("fill", "none")
        .style("opacity", baseOpacity);
    svg.append("line")
        .attr("class", classes + (subtle ? "" : " softmax") + " " + id_class)
        .attr("x1", x(0))
        .attr("y1", y.range()[0] - 2)   // Make the center line stand out slightly by pushing it beyond the rectangle.
        .attr("x2", x(0))
        .attr("y2", y.range()[1] + 2)   // Make the center line stand out slightly by pushing it beyond the rectangle.
        .attr("stroke", black)
        .attr("stroke-width", stroke_width)
        .style("opacity", baseOpacity);
    svg.selectAll(".bar")
        .data(predictions)
        .enter()
            .append("rect")
            .attr("class", classes + (subtle ? "" : " softmax") + " " + id_class)
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
                    return d.colour == null ? "none" : d.colour;
                }

                return "none";
            })
            .style("opacity", baseOpacity)
            .on("mouseover", function(d) {
                d3.selectAll("." + id_class)
                    .transition()
                    .duration(100)
                    .style("opacity", 1.0);
            })
            .on("mouseout", function(d) {
                d3.selectAll("." + id_class)
                    .transition()
                    .duration(50)
                    .style("opacity", baseOpacity);
            });
    svg.selectAll(".bar")
        .data(predictions)
        .enter()
            .append("text")
            .attr("class", classes + (subtle ? "" : " softmax") + " " + id_class)
            .attr("x", function (d) {
                return geometry.x + Math.abs(x(d.value) - x(min)) + 5;
            })
            .attr("y", function (d) {
                return y(d.position) + (y.step() / 2) + 4;
            })
            .style("font-size", "12px")
            .style("opacity", baseOpacity)
            .text(function (d) { return d.label; });
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
                .style("fill", "blue");
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
                .style("fill", "blue");
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

function drawDetail() {
    svg.append("rect")
        .attr("id", "detail-box")
        .attr("class", "detail")
        .attr("x", detail_margin)
        .attr("y", detail_margin)
        .attr("width", total_width - (detail_margin * 2))
        .attr("height", total_height - (detail_margin * 2))
        .attr("stroke", black)
        .attr("stroke-width", 2)
        .attr("fill", "white");
    $("svg").height(total_height);
    drawClose(total_width - detail_margin, detail_margin, (state_width / 4), "detail", function () {
        $(".detail").remove();
        $(".modal").remove();
        input_part = null;
        input_layer = null;
        drawWeightsFromSequence(0);
    });

    if (debug) {
        svg.append("line")
            .attr("class", "detail")
            .attr("x1", (total_width / 2) - 0.5)
            .attr("y1", detail_margin)
            .attr("x2", (total_width / 2) - 0.5)
            .attr("y2", total_height - (detail_margin * 2))
            .attr("stroke", "blue")
            .attr("stroke-width", 1);
        svg.append("line")
            .attr("class", "detail")
            .attr("x1", (detail_margin * 2) + (((total_width / 2) - (detail_margin * 3)) / 2))
            .attr("y1", detail_margin)
            .attr("x2", (total_width / 4) + (detail_margin / 2) - 0.5)
            .attr("y2", total_height - (detail_margin * 2))
            .attr("stroke", "blue")
            .attr("stroke-width", 1);
        svg.append("line")
            .attr("class", "detail")
            .attr("x1", detail_margin)
            .attr("y1", (total_height / 2) - 0.5)
            .attr("x2", total_width - (detail_margin * 2))
            .attr("y2", (total_height / 2) - 0.5)
            .attr("stroke", "blue")
            .attr("stroke-width", 1);
    }

    drawSequenceWheel(true, main_sequence, main_timestep);
    var compareWidth = textWidth("compared to..", 14);
    svg.append("rect")
        .attr("class", "detail compare-button")
        .attr("x", (total_width / 4) + (detail_margin / 2) - (compareWidth / 2))
        .attr("y", (detail_margin * 3) + HEIGHT)
        .attr("width", compareWidth + 5)
        .attr("height", HEIGHT)
        .attr("rx", 5)
        .attr("ry", 5)
        .attr("stroke", "none")
        .attr("fill", light_grey);
    svg.append("text")
        .attr("class", "detail compare-button")
        .attr("x", (total_width / 4) + (detail_margin / 2) - (compareWidth / 2) + 2.5)
        .attr("y", (detail_margin * 3) + HEIGHT + (HEIGHT * .7))
        .style("font-size", "14px")
        .style("fill", black)
        .text("compared to..");
    svg.append("rect")
        .attr("class", "detail compare-button")
        .attr("x", (total_width / 4) + (detail_margin / 2) - (compareWidth / 2))
        .attr("y", (detail_margin * 3) + HEIGHT)
        .attr("width", compareWidth + 5)
        .attr("height", HEIGHT)
        .attr("rx", 5)
        .attr("ry", 5)
        .attr("stroke", dark_grey)
        .attr("stroke-width", 1)
        .attr("fill", "transparent")
        .style("pointer-events", "bounding-box")
        .on("mouseover", function(d) {
            d3.select(this)
                .style("cursor", "pointer");
            d3.select(this)
                .transition()
                .duration(100)
                .attr("stroke-width", 2);
        })
        .on("mouseout", function(d) {
            d3.select(this)
                .style("cursor", "auto");
            d3.select(this)
                .transition()
                .duration(50)
                .attr("stroke-width", 1);
        })
        .on("click", function(d) {
            drawInputModal();
        });
}

function loadDetail(data, placement) {
    $(".detail.load" + (placement == null ? "" : "." + placement)).remove();
    var miniGeometry = {
        x: (total_width / 4) + (detail_margin / 2) - state_width,
        y: (total_height / 2) + (placement == "bottom" ? detail_margin : -((state_height * 2) + detail_margin)),
        width: state_width * 2,
        height: state_height * 2,
    };
    var fullGeometry = {
        x: (total_width / 2) + detail_margin,
        y: detail_margin * 2,
        width: (total_width / 2) - (detail_margin * 3),
        height: total_height - (detail_margin * 4),
    };
    /*if ((h / MEMORY_CHIP_HEIGHT) != (w / MEMORY_CHIP_WIDTH)) {
        throw "chips aren't square (" + (w / MEMORY_CHIP_WIDTH) + ", " + (h / MEMORY_CHIP_HEIGHT) + ")";
    }*/
    if (data.full.vector.length % 2 != 0) {
        throw "vector length (" + data.full.vector.length + ") must be divisible by 2";
    }
    var classes = "detail load" + (placement == null ? "" : " " + placement);
    var linker_suffix = placement == null ? "-top" : "-" + placement;
    drawStateWidget(null, miniGeometry, data.mini.minimum, data.mini.maximum, data.mini.vector, data.mini.colour, data.mini.predictions, classes, MEMORY_CHIP_WIDTH, MEMORY_CHIP_HEIGHT, null, null, null, linker_suffix, null);
    drawStateWidget(null, fullGeometry, data.full.minimum, data.full.maximum, data.full.vector, null, null, classes, 2, data.full.vector.length / 2, null, null, data.back_links, linker_suffix, placement);
}

function drawSequenceWheel(main, sequence, timestep) {
    var x_offset = detail_margin * 2;
    var y_offset = (detail_margin * 2) + (main ? 0 : HEIGHT + detail_margin);
    var width = (total_width / 2) - (detail_margin * 3);
    var height = HEIGHT;
    var type_suffix = main ? "-main" : "-compare";
    $(".wheel" + type_suffix).remove();

    if (debug) {
        svg.append("rect")
            .attr("class", "detail")
            .attr("x", x_offset)
            .attr("y", y_offset)
            .attr("width", width)
            .attr("height", height)
            .attr("stroke", "blue")
            .attr("stroke-width", 1)
            .style("fill", "none");
    }

    var position = 0;
    var datums = sequence.map(word => ({position: position++, word: word}));
    svg.selectAll(".wheel")
        .data(datums)
        .enter()
            .append("text")
            .attr("id", function (d) { return "position-" + d.position + type_suffix; })
            .attr("class", "detail wheel" + type_suffix)
            .attr("x", 0)
            .attr("y", -20)
            .style("font-size", "14px")
            .style("fill", black)
            .text(function(d) { return d.word; })
            .on("mouseover", function(d) {
                d3.select(this)
                    .style("cursor", "pointer");
            })
            .on("mouseout", function(d) {
                d3.select(this)
                    .style("cursor", "auto");
            })
            .on("click", function(d) {
                if (main) {
                    main_timestep = d.position;
                }

                drawSequenceWheel(main, sequence, d.position);
            });
    var center_item_width = textWidth(sequence[timestep], 14);
    var running_width = (center_item_width / 2);
    $("#position-" + timestep + type_suffix)
        .attr("x", x_offset + (width / 2) - running_width)
        .attr("y", y_offset + (height / 2) + (height / 4));
    svg.append("rect")
        .attr("class", "detail wheel" + type_suffix)
        .attr("x", x_offset + (width / 2) - running_width - 2.5)
        .attr("y", y_offset)
        .attr("width", center_item_width + 5)
        .attr("height", height)
        .attr("stroke", "black")
        .attr("stroke-width", 2)
        .style("fill", "none");
    var space_width = textWidth("&nbsp;", 14) + 1;
    for (var i = 1; i <= timestep; i++) {
        var item_width = textWidth(sequence[timestep - i], 14);
        running_width += item_width + space_width;
        $("#position-" + (timestep - i) + type_suffix)
            .attr("x", x_offset + (width / 2) - running_width)
            .attr("y", y_offset + (height / 2) + (height / 4))
            .css("opacity", 1.0 - (Math.max(i - 3, 0) * .2));
    }
    running_width = (center_item_width / 2);
    for (var i = 1; i <= sequence.length; i++) {
        running_width += space_width;
        $("#position-" + (timestep + i) + type_suffix)
            .attr("x", x_offset + (width / 2) + running_width)
            .attr("y", y_offset + (height / 2) + (height / 4))
            .css("opacity", 1.0 - (Math.max(i - 3, 0) * .2));
        var item_width = textWidth(sequence[timestep + i], 14);
        running_width += item_width;
    }

    // Load the data based off the center item.
    var slice = sequence.slice(0, timestep + 1);
    var distance = sequence.length - timestep - 1;
    console.log("Drawing detail for " + (slice.length - 1) + " @" + distance + ": " + slice);
    var layerParameter = input_layer == null ? "" : "&layer=" + input_layer;
    var placement = main ? (compare_sequence.length == 0 ? null : "top") : "bottom";
    d3.json("weight-detail?distance=" + distance + "&part=" + input_part + layerParameter + "&" + slice.map(s => "sequence=" + encodeURI(s)).join("&"))
        .get(function (error, data) { loadDetail(data, placement); });
}

function drawOpen(x_offset, y_offset, radius, items_class, callback) {
    var id_class = "open-" + Math.random().toString(36).substring(5);
    var stroke_width = 1;
    svg.append("circle")
        .attr("class", items_class)
        .attr("cx", x_offset)
        .attr("cy", y_offset)
        .attr("r", radius)
        .attr("stroke", "none")
        .attr("stroke-width", stroke_width)
        .attr("fill", light_grey)
    svg.append("line")
        .attr("class", items_class + " " + id_class)
        .attr("x1", x_offset - (radius / 2))
        .attr("y1", y_offset)
        .attr("x2", x_offset + (radius / 2))
        .attr("y2", y_offset)
        .attr("stroke", dark_grey)
        .attr("stroke-width", stroke_width);
    svg.append("line")
        .attr("class", items_class + " " + id_class)
        .attr("x1", x_offset)
        .attr("y1", y_offset - (radius / 2))
        .attr("x2", x_offset)
        .attr("y2", y_offset + (radius / 2))
        .attr("stroke", dark_grey)
        .attr("stroke-width", stroke_width);
    svg.append("circle")
        .attr("class", items_class + " " + id_class)
        .attr("cx", x_offset)
        .attr("cy", y_offset)
        .attr("r", radius)
        .attr("stroke", dark_grey)
        .attr("stroke-width", stroke_width)
        .attr("fill", "transparent")
        .style("pointer-events", "bounding-box")
        .on("mouseover", function(d) {
            d3.select(this)
                .style("cursor", "pointer");
            d3.selectAll("." + id_class)
                .transition()
                .duration(100)
                .attr("stroke-width", stroke_width * 2);
        })
        .on("mouseout", function(d) {
            d3.select(this)
                .style("cursor", "auto");
            d3.selectAll("." + id_class)
                .transition()
                .duration(50)
                .attr("stroke-width", stroke_width);
        })
        .on("click", function(d) {
            callback();
        });
}

function drawClose(x_offset, y_offset, radius, items_class, callback) {
    var stroke_width = 1;
    var id_class = "open-" + Math.random().toString(36).substring(5);
    svg.append("circle")
        .attr("class", items_class)
        .attr("cx", x_offset)
        .attr("cy", y_offset)
        .attr("r", radius)
        .attr("stroke", "none")
        .attr("stroke-width", stroke_width)
        .attr("fill", light_grey)
    svg.append("line")
        .attr("class", items_class + " " + id_class)
        .attr("x1", x_offset - (radius / 2))
        .attr("y1", y_offset)
        .attr("x2", x_offset + (radius / 2))
        .attr("y2", y_offset)
        .attr("stroke", dark_grey)
        .attr("stroke-width", stroke_width);
    svg.append("circle")
        .attr("class", items_class + " " + id_class)
        .attr("cx", x_offset)
        .attr("cy", y_offset)
        .attr("r", radius)
        .attr("stroke", dark_grey)
        .attr("stroke-width", 1)
        .attr("fill", "transparent")
        .style("pointer-events", "bounding-box")
        .on("mouseover", function(d) {
            d3.select(this)
                .style("cursor", "pointer");
            d3.selectAll("." + id_class)
                .transition()
                .duration(100)
                .attr("stroke-width", stroke_width * 2);
        })
        .on("mouseout", function(d) {
            d3.select(this)
                .style("cursor", "auto");
            d3.selectAll("." + id_class)
                .transition()
                .duration(50)
                .attr("stroke-width", stroke_width);
        })
        .on("click", function(d) {
            callback();
        });
}

function drawAutocomplete(timestep, words) {
    var x_offset = x_margin;
    var y_offset = y_margin + (timestep * layer_height)
    var focus = null;

    svg.append("foreignObject")
        .attr("transform", "translate(" + x_offset + "," + (y_offset + state_height - (HEIGHT / 2) - 1) + ")")
        .attr("width", input_width)
        .attr("height", HEIGHT)
        .append("xhtml:div")
        .attr("id", "autocomplete-" + timestep);
    var autocomplete = $("#autocomplete-" + timestep);
    autocomplete.append("<input class=':focus'/>");
    autocomplete.find("input").focus();
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

            if (timestep >= main_sequence.length) {
                main_sequence.push(textContent);
                d3.json("words")
                    .get(function (error, data) { drawAutocomplete(timestep + 1, data); });
            } else {
                main_sequence[timestep] = textContent;
            }

            drawWeightsFromSequence(0);
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
                    var tail = main_sequence.length;
                    main_sequence = main_sequence.slice(0, timestep);

                    for (var s = timestep; s <= tail; s++) {
                        if (s != timestep) {
                            $("#autocomplete-" + s).remove();
                        }

                        $(".timestep-" + s).remove();
                    }
                } else {
                    if (timestep >= main_sequence.length) {
                        main_sequence.push(textContent);
                        d3.json("words")
                            .get(function (error, data) { drawAutocomplete(timestep + 1, data); });
                    } else {
                        main_sequence[timestep] = textContent;
                    }
                }

                drawWeightsFromSequence(0);
            }
        }

        autocomplete.find(".autocomplete-active").removeClass("autocomplete-active");

        if (focus >= 0) {
            autocomplete.find(".autocomplete-option:eq(" + focus + ")").addClass("autocomplete-active");
        }
    });
}

function drawWeightsFromSequence(timestep) {
    console.log("Full sequence: " + main_sequence);

    for (var s = timestep; s < main_sequence.length; s++) {
        var slice = main_sequence.slice(0, s + 1);
        var distance = main_sequence.length - s - 1;
        console.log("Drawing sequence for " + (slice.length - 1) + " @" + distance + ": " + slice);
        d3.json("weights?distance=" + distance + "&" + slice.map(s => "sequence=" + encodeURI(s)).join("&"))
            .get(function (error, data) { drawTimestep(slice.length - 1, data); });
    }
}

function drawInputModal() {
    var width = (total_width / 2);
    var height = (total_height / 2);
    var x_offset = (total_width - width) / 2;
    var y_offset = (total_height - height) / 2;
    svg.append("rect")
        .attr("class", "modal")
        .attr("x", x_offset)
        .attr("y", y_offset)
        .attr("width", width)
        .attr("height", height)
        .attr("stroke", black)
        .attr("stroke-width", 2)
        .attr("fill", "white");
    $("svg").height(total_height);
    drawClose(x_offset + width, y_offset, (state_width / 4), "modal", function () {
        $(".modal").remove();
    });
    svg.append("foreignObject")
        .attr("class", "modal")
        .attr("transform", "translate(" + (x_offset + detail_margin) + "," + (y_offset + detail_margin) + ")")
        .attr("width", width - (detail_margin * 2))
        .attr("height", HEIGHT)
        .append("xhtml:div")
        .attr("id", "sequence-inputter");
    var sequenceInputter = $("#sequence-inputter");
    sequenceInputter.append("<input class=':focus'/>");
    sequenceInputter.on("keydown", function(e) {
        // Enter key
        if (e.keyCode == 13) {
            compare_sequence = sequenceInputter.find("input")
                .val()
                .split(" ");
            sequenceInputter.find("input")
                .val("");
            $(".modal").remove();
            $(".compare-button").remove();
            $(".detail.load").remove();
            drawSequenceWheel(true, main_sequence, main_timestep);
            drawSequenceWheel(false, compare_sequence, 0);
        }
    });
}

function getGeometry(timestep, part, layer) {
    var x_offset = (x_margin * 2) + input_width;
    var y_offset = y_margin + (timestep * layer_height);
    var layer_offset = layer * state_width * 14;
    var b;

    switch (part) {
        case "embedding":
            b = {x: x_offset + state_width, y: y_offset + (state_height * 3 / 4)};
            break;
        case "cell_previouses":
            b = {x: x_offset + (state_width * 9 / 2) + layer_offset, y: y_offset};
            break;
        /*case "forget_gate":
            b = {x: x_offset + (w * 3) + (w / 2) + layer_offset, y: y_offset + (h * 1 / 2) + (operator_height / 2), height: operand_height};
            break;*/
        case "forgets":
            b = {x: x_offset + (state_width * 8) + layer_offset, y: y_offset};
            break;
        case "input_hats":
            b = {x: x_offset + (state_width * 9 / 2) + layer_offset, y: y_offset + (state_height * 3 / 2)};
            break;
        /*case "remember_gate":
            b = {x: x_offset + (w * 7) + (w / 2) + layer_offset, y: y_offset + (h * 3 / 2) + (operator_height / 2), height: operand_height};
            break;*/
        case "remembers":
            b = {x: x_offset + (state_width * 8) + layer_offset, y: y_offset + (state_height * 3 / 2)};
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
        case "cell_hats":
            b = {x: x_offset + (state_width * 11) + (state_width / 2) + layer_offset, y: y_offset + (state_height * 3 / 4)};
            break;
        /*case "output_gate":
            b = {x: x_offset + (w * 15) + (w / 2) + layer_offset, y: y_offset + h + (operator_height / 2), height: operand_height};
            break;*/
        case "outputs":
            b = {x: x_offset + (state_width * 15) + layer_offset, y: y_offset + (state_height * 3 / 4)};
            break;
        case "softmax":
            // For the 2 layers v
            b = {x: x_offset + (2 * state_width * 17) + (state_width * 3 / 2), y: y_offset + (state_height * 3 / 4)};
            break;
        default:
            return null;
    }

    return Object.assign({}, {width: state_width, height: state_height}, b);
}

function textWidth(text, fontSize) {
    var temporaryDiv = document.createElement("div");
    document.body.appendChild(temporaryDiv);
    temporaryDiv.style.fontSize = "" + fontSize + "px";
    temporaryDiv.style.position = "absolute";
    temporaryDiv.style.left = -100;
    temporaryDiv.style.top = -100;
    temporaryDiv.innerHTML = text;
    var width = temporaryDiv.clientWidth;
    document.body.removeChild(temporaryDiv);
    temporaryDiv = null;
    return width;
}


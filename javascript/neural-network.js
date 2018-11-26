
var total_width = 1200;
var layer_height = 200;
var input_width = 100;
var margin = 25;
var w = 30;
var h = layer_height / 3.0;
var operand_height = (h * 2.0 / 5.0);
var operator_height = (h - (operand_height * 2));
var black = "black";
var light_grey = "#dee0e2";
var debug = window.location.hash.substring(1) == "debug";
var svg = null;
var sequence = [];

$(document).ready(function () {
    svg = d3.select('body').append('svg')
        .style('position', 'absolute')
        .style('top', 0)
        .style('left', 0)
        .style('width', total_width * 2)
        .style('height', layer_height * 5);

    d3.json("words")
        .get(function (error, data) { drawAutocomplete(0, data); });

   drawGate(25,150,100);
   
   svg.append("rect")
    .attr("width", 2400)
    .attr("height", 1000)
    .style("fill", "none")
    .style("pointer-events", "all")
    .call(d3.zoom()
       .scaleExtent([1, 16])
        .on("zoom", zoomed));

});

function zoomed() {
  svg.attr("transform", d3.event.transform);
}

function drawTimestep(fake_timestep, data) {
    console.log("Timestep (fake, actual): (" + fake_timestep + ", " + data.timestep + ")");
    console.log(data);
    var timestep = data.timestep;
    $(".timestep-" + timestep).remove();

    var x_offset = (margin * 2) + input_width;
    var y_offset = margin + (timestep * layer_height);
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
    drawWeightVector(getGeometry(timestep, "embedding"), data.embedding);

    // Draw units
    for (var u = 0; u < data.units.length; u++) {
        var unit_offset = u * w * 16;

        if (timestep > 0) {
            drawVline(timestep, x_offset + (w * 13) + (w / 2) + unit_offset, margin + ((timestep - 1) * layer_height) + (h * 3 / 2),
                x_offset + (w * 4) + unit_offset, y_offset);
        }

        drawWeightVector(getGeometry(timestep, "cell_previous_hat", u), data.units[u].cell_previous_hat);
        drawMultiplication(timestep, x_offset + (w * 3) + (w) - (operator_height / 2) + unit_offset, y_offset + (h * 1 / 2) - (operator_height / 2), operator_height);
        drawWeightVector(getGeometry(timestep, "forget_gate", u), data.units[u].forget_gate);
        drawHline(timestep, x_offset + (w * 4) + (operator_height / 2) + unit_offset, y_offset + (h / 2),
            x_offset + (w * 5) + unit_offset, y_offset + (h / 2));
        drawWeightVector(getGeometry(timestep, "forget", u), data.units[u].forget);
        drawHline(timestep, x_offset + (w * 2) + unit_offset, y_offset + h,
            x_offset + (w * 7) + (w / 2) + unit_offset, y_offset + h + (operand_height / 2),
            x_offset + (w * 2) + unit_offset + ((w * 3 / 2) / 2), y_offset + h + (operand_height / 4));

        if (timestep > 0) {
            drawVline(timestep, x_offset + (w * 17) + (w / 2) + unit_offset, margin + ((timestep - 1) * layer_height) + (h * 3 / 2),
                x_offset + (w * 8) + unit_offset, y_offset + (h ));
        }

        drawWeightVector(getGeometry(timestep, "input_hat", u), data.units[u].input_hat);
        drawMultiplication(timestep, x_offset + (w * 7) + (w) - (operator_height / 2) + unit_offset, y_offset + (h * 3 / 2) - (operator_height / 2), operator_height);
        drawWeightVector(getGeometry(timestep, "remember_gate", u), data.units[u].remember_gate);
        drawHline(timestep, x_offset + (w * 8) + (operator_height / 2) + unit_offset, y_offset + (h * 3 / 2),
            x_offset + (w * 9) + unit_offset, y_offset + (h * 3 / 2));
        drawWeightVector(getGeometry(timestep, "remember", u), data.units[u].remember);
        drawHline(timestep, x_offset + (w * 6) + unit_offset, y_offset + (h / 2),
            x_offset + (w * 11) + (w / 2) + unit_offset, y_offset + (h / 2) + (operand_height / 2));
        drawWeightVector(getGeometry(timestep, "forget_hat", u), data.units[u].forget);
        drawHline(timestep, x_offset + (w * 10) + unit_offset, y_offset + (h * 3 / 2),
            x_offset + (w * 11) + (w / 2) + unit_offset, y_offset + h + (operand_height / 2) + (operator_height / 2));
        drawAddition(timestep, x_offset + (w * 11) + (w) - (operator_height / 2) + unit_offset, y_offset + (h) - (operator_height / 2), operator_height);
        drawWeightVector(getGeometry(timestep, "remember_hat", u), data.units[u].remember);
        drawHline(timestep, x_offset + (w * 12) + (operator_height / 2) + unit_offset, y_offset + (h * 2 / 2),
            x_offset + (w * 13) + unit_offset, y_offset + (h * 2 / 2));
        drawWeightVector(getGeometry(timestep, "cell", u), data.units[u].cell);
        drawHline(timestep, x_offset + (w * 14) + unit_offset, y_offset + h,
            x_offset + (w * 15) + (w / 2) + unit_offset, y_offset + (h / 2) + (operand_height / 2));
        drawWeightVector(getGeometry(timestep, "cell_hat", u), data.units[u].cell_hat);
        drawMultiplication(timestep, x_offset + (w * 15) + (w) - (operator_height / 2) + unit_offset, y_offset + (h) - (operator_height / 2), operator_height);
        drawWeightVector(getGeometry(timestep, "output_gate", u), data.units[u].output_gate);
        drawHline(timestep, x_offset + (w * 16) + (operator_height / 2) + unit_offset, y_offset + (h * 2 / 2),
            x_offset + (w * 17) + unit_offset, y_offset + (h * 2 / 2));
        drawWeightVector(getGeometry(timestep, "output", u), data.units[u].output);
    }

    // Draw softmax
    drawHline(timestep, x_offset + (data.units.length * w * 17), y_offset + (h * 2 / 2),
        x_offset + (data.units.length * w * 17) + (w * 3 / 2), y_offset + (h * 2 / 2));
    drawLabelWeightVector(getGeometry(timestep, "softmax"), data.softmax);
}

function drawWeightVector(geometry, wv) {
    drawWeightWidget(geometry, wv.minimum, wv.maximum, wv.vector, wv.colour);
}

function drawLabelWeightVector(geometry, lwv) {
    drawWeightWidget(geometry, lwv.minimum, lwv.maximum, lwv.vector, "none");
}

function drawWeightWidget(geometry, min, max, vector, colour) {
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
        .attr("fill", "none");
    svg.append("line")
        .attr("class", "timestep-" + geometry.timestep)
        .attr("x1", x(0))
        .attr("y1", y.range()[0] - 2)   // Make the center line stand out slightly by pushing it beyond the rectangle.
        .attr("x2", x(0))
        .attr("y2", y.range()[1] + 2)   // Make the center line stand out slightly by pushing it beyond the rectangle.
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
    // append the rectangles for the bar chart
    //var weights = "[" + vector.map(v => v.value).join(",") + "]";
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
            .attr("fill", function(d) { return colour; });
            // Using the colour 'none' results in the inner portion of the rectange not being clickable.
            // So we fill with the background colour (white) to make it clickable.
            //.attr("fill", function(d) { return colour == "none" ? "white" : colour; });
    svg.selectAll(".bar")
        .data(vector)
        .enter()
            .append("rect")
            .attr("id", function(d) { return "hoverbar-" + geometry.timestep + "-" + geometry.name + "-" + d.position; })
            .attr("class", "timestep-" + geometry.timestep)
            .attr("x", geometry.x + 1)
            .attr("y", function(d) { return y(d.position) + 1; })
            .attr("width", geometry.width - 2)
            .attr("height", y.bandwidth() - 2)
            .attr("stroke", black)
            .attr("stroke-width", 2)
            .attr("fill", black)
            .style("opacity", 0)
            .on("mouseover", function(d) {
                if (geometry.name != "embedding" && (geometry.timestep > 0 || !geometry.name.startsWith("cell_previous_hat"))) {
                    d3.select("#hoverbar-" + geometry.timestep + "-" + geometry.name + "-" + d.position)
                        .style("opacity", 0.5);
                }
            })
            .on("mouseout", function(d) {
                if (geometry.name != "embedding" && (geometry.timestep > 0 || !geometry.name.startsWith("cell_previous_hat"))) {
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
                if (geometry.name != "embedding" && (geometry.timestep > 0 || !geometry.name.startsWith("cell_previous_hat"))) {
                    var slice = sequence.slice(0, geometry.timestep + 1);
                    console.log(geometry.name);
                    d3.json("weight-explain?" + slice.map(s => "sequence=" + encodeURI(s)).join("&") + "&name=" + geometry.name + "&column=" + d.column)
                        .get(function (error, we) {
                            drawExplain(geometry.timestep, source, we);
                        });
                }
            });
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
            .text(function (d) { return d.label; });
}

function drawExplain(timestep, source, we) {
    console.log(we);
    $(".explain").remove();
    svg.append("rect")
        .attr("class", "explain")
        .attr("x", source.x)
        .attr("y", source.y)
        .attr("width", source.width)
        .attr("height", source.height)
        .attr("stroke-width", 2)
        .attr("stroke", black)
        .attr("fill", "none")
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
        .attr("x", geometry.x + 1)
        .attr("y", geometry.y + 1)
        .attr("width", geometry.width - 2)
        .attr("height", geometry.height - 2)
        .style("opacity", 0.5)
        .attr("stroke", black)
        .attr("stroke-width", 2)
        .attr("fill", "none");
    svg.append("line")
        .attr("class", "timestep-" + geometry.timestep + " explain")
        .attr("x1", x(0))
        .attr("y1", y.range()[0] - 2)   // Make the center line stand out slightly by pushing it beyond the rectangle.
        .attr("x2", x(0))
        .attr("y2", y.range()[1] + 2)   // Make the center line stand out slightly by pushing it beyond the rectangle.
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
            .style("fill", "none");
}

function drawOperatorCircle(timestep, x_offset, y_offset, size) {
    svg.append("circle")
        .attr("class", "timestep-" + timestep)
        .attr("cx", x_offset + (size / 2))
        .attr("cy", y_offset + (size / 2))
        .attr("r", (size / 2) - 0.5)
        .attr("stroke", black)
        .attr("stroke-width", 1)
        .attr("fill", light_grey);
}

function drawAddition(timestep, x_offset, y_offset, size) {
    drawOperatorCircle(timestep, x_offset, y_offset, size);
    var stroke_width = size / 10;
    var stroke_length = size / 2;
    svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset + ((size - stroke_length) / 2))
        .attr("y1", y_offset + (size / 2))
        .attr("x2", x_offset + ((size - stroke_length) / 2) + stroke_length)
        .attr("y2", y_offset + (size / 2))
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
    svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset + (size / 2))
        .attr("y1", y_offset + ((size - stroke_length) / 2))
        .attr("x2", x_offset + (size / 2))
        .attr("y2", y_offset + ((size - stroke_length) / 2) + stroke_length)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
}

function drawMultiplication(timestep, x_offset, y_offset, size) {
    drawOperatorCircle(timestep, x_offset, y_offset, size);
    var stroke_width = size / 10;
    var stroke_length = size / 2;
    var qq = Math.sqrt((stroke_length**2) / 2);
    svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset + ((size - qq) / 2))
        .attr("y1", y_offset + ((size - qq) / 2))
        .attr("x2", x_offset + ((size - qq) / 2) + qq)
        .attr("y2", y_offset + ((size - qq) / 2) + qq)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
    svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset + ((size - qq) / 2))
        .attr("y1", y_offset + ((size - qq) / 2) + qq)
        .attr("x2", x_offset + ((size - qq) / 2) + qq)
        .attr("y2", y_offset + ((size - qq) / 2))
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
}

function drawEquals(timestep, x_offset, y_offset, size) {
    drawOperatorCircle(timestep, x_offset, y_offset, size);
    var stroke_width = size / 10;
    var stroke_length = size / 2;
    svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset + ((size - stroke_length) / 2))
        .attr("y1", y_offset + (size / 2) - (stroke_width * 1))
        .attr("x2", x_offset + ((size - stroke_length) / 2) + stroke_length)
        .attr("y2", y_offset + (size / 2) - (stroke_width * 1))
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
    svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset + ((size - stroke_length) / 2))
        .attr("y1", y_offset + (size / 2) + (stroke_width * 1))
        .attr("x2", x_offset + ((size - stroke_length) / 2) + stroke_length)
        .attr("y2", y_offset + (size / 2) + (stroke_width * 1))
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
}

function drawGate(x_offset, y_offset, size) {
	
    var stroke_width = size/50;
    
    //left vertical bar
    svg.append("rect")
        .attr("x", x_offset)
        .attr("y", y_offset+size/8)
        .attr("width", size/10)
        .attr("height", size-size/8)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width)
        .attr("fill", "#FFF");

    //left circle
    svg.append("circle")
        .attr("cx", x_offset+size*2/39)
        .attr("cy", y_offset+size/18)
        .attr("r", size/16)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width)
        .attr("fill", "#FFF");


    //right vertical bar
    svg.append("rect")
        .attr("x", x_offset+size*3/2-size/8)
        .attr("y", y_offset+size/8)
        .attr("width", size/10)
        .attr("height", size-size/8)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width)
        .attr("fill", "#FFF");

    //right circle
    svg.append("circle")
        .attr("cx", x_offset+size*3/2-size*2/27)
        .attr("cy", y_offset+size/18)
        .attr("r", size/16)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width)
        .attr("fill", "#FFF"); 

    //left door
    svg.append("line")
	 .attr("x1", x_offset+size*13/20)
        .attr("y1", y_offset)
        .attr("x2", x_offset+size*13/20)
        .attr("y2", y_offset+size-size/6)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);

   svg.append("line")
	 .attr("x1", x_offset+size*13/20)
        .attr("y1", y_offset)
        .attr("x2", x_offset+size/10)
        .attr("y2", y_offset+size/6)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);

   svg.append("line")
	 .attr("x1", x_offset+size*13/20)
        .attr("y1", y_offset+size-size/6)
        .attr("x2", x_offset+size/10)
        .attr("y2", y_offset+size-size/8)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);

   //right door
   svg.append("line")
	 .attr("x1", x_offset+size*4/5)
        .attr("y1", y_offset)
        .attr("x2", x_offset+size*4/5)
        .attr("y2", y_offset+size-size/6)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);

   svg.append("line")
	 .attr("x1", x_offset+size*4/5)
        .attr("y1", y_offset)
        .attr("x2", x_offset+size*3/2-size/8)
        .attr("y2", y_offset+size/6)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);

   svg.append("line")
	 .attr("x1", x_offset+size*4/5)
        .attr("y1", y_offset+size-size/6)
        .attr("x2", x_offset+size*3/2-size/8)
        .attr("y2", y_offset+size-size/8)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);

}

function drawAutocomplete(timestep, words) {
    var x_offset = 25;
    var y_offset = 25 + (timestep * layer_height)
    var height = 20;
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
            if (words[i].substr(0, value.length).toUpperCase() === value.toUpperCase()) {
                autocomplete.append("<div class='autocomplete-option'>" + words[i] + "</div>");
            }
        }

        $(".autocomplete-option").on("click", function(e) {
            autocomplete.find(".autocomplete-option").remove();
            autocomplete.find("input").val(e.target.textContent);

            if (timestep >= sequence.length) {
                sequence.push(e.target.textContent);
                d3.json("words")
                    .get(function (error, data) { drawAutocomplete(timestep + 1, data); });
            } else {
                sequence[timestep] = e.target.textContent;
            }

            console.log("Full sequence: " + sequence);

            for (var s = timestep; s < sequence.length; s++) {
                var slice = sequence.slice(0, s + 1);
                console.log("Drawing sequence for " + (slice.length - 1) + ": " + slice);
                d3.json("weights?" + slice.map(s => "sequence=" + encodeURI(s)).join("&"))
                    .get(function (error, data) { drawTimestep(slice.length - 1, data); });
            }
        });
    })
    .on("keydown", function(e) {
        var options = autocomplete.find(".autocomplete-option").length;

        // Down key
        if (e.keyCode === 40) {
            if (focus == options - 1) {
                focus = 0;
            } else {
                focus += 1;
            }
        }
        // Up key
        else if (e.keyCode === 38) {
            if (focus == 0) {
                focus = options - 1;
            } else {
                focus -= 1;
            }
        }
        // Enter key
        else if (e.keyCode == 13) {
            autocomplete.find(".autocomplete-active").click();
        }

        autocomplete.find(".autocomplete-active").removeClass("autocomplete-active");
        autocomplete.find(".autocomplete-option:eq(" + focus + ")").addClass("autocomplete-active");
    });
}

function getGeometry(timestep, name, layer) {
    var x_offset = (margin * 2) + input_width;
    var y_offset = margin + (timestep * layer_height);
    var layer_offset = layer * w * 16;
    var a = {width: w, timestep: timestep, name: name + (layer == null ? "" : "-" + layer)};
    var b;

    switch (name) {
        case "embedding":
            b = {x: x_offset + w, y: y_offset + (h / 2), height: h};
            break;
        case "cell_previous_hat":
            b = {x: x_offset + (w * 3) + (w / 2) + layer_offset, y: y_offset, height: operand_height};
            break;
        case "forget_gate":
            b = {x: x_offset + (w * 3) + (w / 2) + layer_offset, y: y_offset + (h * 1 / 2) + (operator_height / 2), height: operand_height};
            break;
        case "forget":
            b = {x: x_offset + (w * 5) + layer_offset, y: y_offset, height: h};
            break;
        case "input_hat":
            b = {x: x_offset + (w * 7) + (w / 2) + layer_offset, y: y_offset + h, height: operand_height};
            break;
        case "remember_gate":
            b = {x: x_offset + (w * 7) + (w / 2) + layer_offset, y: y_offset + (h * 3 / 2) + (operator_height / 2), height: operand_height};
            break;
        case "remember":
            b = {x: x_offset + (w * 9) + layer_offset, y: y_offset + h, height: h};
            break;
        case "forget_hat":
            b = {x: x_offset + (w * 11) + (w / 2) + layer_offset, y: y_offset + (h * 1 / 2), height: operand_height};
            break;
        case "remember_hat":
            b = {x: x_offset + (w * 11) + (w / 2) + layer_offset, y: y_offset + (h * 2 / 2) + (operator_height / 2), height: operand_height};
            break;
        case "cell":
            b = {x: x_offset + (w * 13) + layer_offset, y: y_offset + (h * 1 / 2), height: h};
            break;
        case "cell_hat":
            b = {x: x_offset + (w * 15) + (w / 2) + layer_offset, y: y_offset + (h / 2), height: operand_height};
            break;
        case "output_gate":
            b = {x: x_offset + (w * 15) + (w / 2) + layer_offset, y: y_offset + h + (operator_height / 2), height: operand_height};
            break;
        case "output":
            b = {x: x_offset + (w * 17) + layer_offset, y: y_offset + (h / 2), height: h};
            break;
        case "softmax":
            // For the 2 layers v
            b = {x: x_offset + (2 * w * 17) + (w * 3 / 2), y: y_offset + (h / 2), height: h};
            break;
    }

    return Object.assign({}, a, b);
}


var total_width = 1200;
var layer_height = 200;
var input_width = 100;
var margin = 25;
var w = 30;
var h = layer_height / 3.0;
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
});

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
    drawWeightVector(timestep, data.embedding, "embedding",
        x_offset + (w), y_offset + (h / 2),
        w, h);

    // Draw units
    for (var u = 0; u < data.units.length; u++) {
        var unit_offset = u * w * 16;

        if (timestep > 0) {
            drawVline(timestep, x_offset + (w * 13) + (w / 2) + unit_offset, margin + ((timestep - 1) * layer_height) + (h * 3 / 2),
                x_offset + (w * 4) + unit_offset, y_offset);
        }

        drawWeightVector(timestep, data.units[u].cell_previous, "cell_previous-" + u,
            x_offset + (w * 3) + (w / 2) + unit_offset, y_offset,
            w, operand_height);
        drawMultiplication(timestep, x_offset + (w * 3) + (w) - (operator_height / 2) + unit_offset, y_offset + (h * 1 / 2) - (operator_height / 2), operator_height);
        drawWeightVector(timestep, data.units[u].forget_gate, "forget_gate-" + u,
            x_offset + (w * 3) + (w / 2) + unit_offset, y_offset + (h * 1 / 2) + (operator_height / 2),
            w, operand_height);
        drawHline(timestep, x_offset + (w * 4) + (operator_height / 2) + unit_offset, y_offset + (h / 2),
            x_offset + (w * 5) + unit_offset, y_offset + (h / 2));
        drawWeightVector(timestep, data.units[u].forget, "forget-" + u,
            x_offset + (w * 5) + unit_offset, y_offset,
            w, h);
        drawHline(timestep, x_offset + (w * 2) + unit_offset, y_offset + h,
            x_offset + (w * 7) + (w / 2) + unit_offset, y_offset + h + (operand_height / 2),
            x_offset + (w * 2) + unit_offset + ((w * 3 / 2) / 2), y_offset + h + (operand_height / 4));

        if (timestep > 0) {
            drawVline(timestep, x_offset + (w * 17) + (w / 2) + unit_offset, margin + ((timestep - 1) * layer_height) + (h * 3 / 2),
                x_offset + (w * 8) + unit_offset, y_offset + (h ));
        }

        drawWeightVector(timestep, data.units[u].input_hat, "input_hat-" + u,
            x_offset + (w * 7) + (w / 2) + unit_offset, y_offset + h,
            w, operand_height);
        drawMultiplication(timestep, x_offset + (w * 7) + (w) - (operator_height / 2) + unit_offset, y_offset + (h * 3 / 2) - (operator_height / 2), operator_height);
        drawWeightVector(timestep, data.units[u].remember_gate, "remember_gate-" + u,
            x_offset + (w * 7) + (w / 2) + unit_offset, y_offset + (h * 3 / 2) + (operator_height / 2),
            w, operand_height);
        drawHline(timestep, x_offset + (w * 8) + (operator_height / 2) + unit_offset, y_offset + (h * 3 / 2),
            x_offset + (w * 9) + unit_offset, y_offset + (h * 3 / 2));
        drawWeightVector(timestep, data.units[u].remember, "remember-" + u,
            x_offset + (w * 9) + unit_offset, y_offset + h,
            w, h);
        drawHline(timestep, x_offset + (w * 6) + unit_offset, y_offset + (h / 2),
            x_offset + (w * 11) + (w / 2) + unit_offset, y_offset + (h / 2) + (operand_height / 2));
        drawWeightVector(timestep, data.units[u].forget, "forget-" + u,
            x_offset + (w * 11) + (w / 2) + unit_offset, y_offset + (h * 1 / 2),
            w, operand_height);
        drawHline(timestep, x_offset + (w * 10) + unit_offset, y_offset + (h * 3 / 2),
            x_offset + (w * 11) + (w / 2) + unit_offset, y_offset + h + (operand_height / 2) + (operator_height / 2));
        drawAddition(timestep, x_offset + (w * 11) + (w) - (operator_height / 2) + unit_offset, y_offset + (h) - (operator_height / 2), operator_height);
        drawWeightVector(timestep, data.units[u].remember, "remember-" + u,
            x_offset + (w * 11) + (w / 2) + unit_offset, y_offset + (h * 2 / 2) + (operator_height / 2),
            w, operand_height);
        drawHline(timestep, x_offset + (w * 12) + (operator_height / 2) + unit_offset, y_offset + (h * 2 / 2),
            x_offset + (w * 13) + unit_offset, y_offset + (h * 2 / 2));
        drawWeightVector(timestep, data.units[u].cell, "cell-" + u,
            x_offset + (w * 13) + unit_offset, y_offset + (h * 1 / 2),
            w, h);
        drawHline(timestep, x_offset + (w * 14) + unit_offset, y_offset + h,
            x_offset + (w * 15) + (w / 2) + unit_offset, y_offset + (h / 2) + (operand_height / 2));
        drawWeightVector(timestep, data.units[u].cell_hat, "cell_hat-" + u,
            x_offset + (w * 15) + (w / 2) + unit_offset, y_offset + (h / 2),
            w, operand_height);
        drawMultiplication(timestep, x_offset + (w * 15) + (w) - (operator_height / 2) + unit_offset, y_offset + (h) - (operator_height / 2), operator_height);
        drawWeightVector(timestep, data.units[u].output_gate, "output_gate-" + u,
            x_offset + (w * 15) + (w / 2) + unit_offset, y_offset + h + (operator_height / 2),
            w, operand_height);
        drawHline(timestep, x_offset + (w * 16) + (operator_height / 2) + unit_offset, y_offset + (h * 2 / 2),
            x_offset + (w * 17) + unit_offset, y_offset + (h * 2 / 2));
        drawWeightVector(timestep, data.units[u].output, "output-" + u,
            x_offset + (w * 17) + unit_offset, y_offset + (h / 2),
            w, h);
    }

    // Draw softmax
    drawHline(timestep, x_offset + (data.units.length * w * 17), y_offset + (h * 2 / 2),
        x_offset + (data.units.length * w * 17) + (w * 3 / 2), y_offset + (h * 2 / 2));
    drawLabelWeightVector(timestep, data.softmax, "softmax", x_offset + (data.units.length * w * 17) + (w * 3 / 2), y_offset + (h / 2), w, h);
}

function drawWeightVector(timestep, weight, name, x_offset, y_offset, width, height) {
    drawWeightWidget(timestep, name, x_offset, y_offset, width, height, weight.minimum, weight.maximum, weight.vector, weight.colour);
}

function drawLabelWeightVector(timestep, label_weight, name, x_offset, y_offset, width, height) {
    drawWeightWidget(timestep, name, x_offset, y_offset, width, height, label_weight.minimum, label_weight.maximum, label_weight.vector, "none");
}

function drawWeightWidget(timestep, name, x_offset, y_offset, width, height, min, max, vector, colour) {
    if (min >= max) {
        throw "min " + min + " cannot exceed max " + max;
    }

    if (min > 0) {
        throw "min " + min + " cannot be greater than 0";
    }

    if (max < 0) {
        throw "max " + max + " cannot be less than 0";
    }

    var found_min = d3.min(vector, function(d) { return d.life_expect; });
    if (found_min > min) {
        throw "found value " + found_min + " exceeding min " + min;
    }

    var found_max = d3.max(vector, function(d) { return d.life_expect; });
    if (found_max > max) {
        throw "found value " + found_max + " exceeding max " + max;
    }

    var stroke_width = 1;

    var y = d3.scaleBand()
        .domain(vector.map(function (d) { return d.position; }))
        .range([y_offset + (stroke_width / 2.0), y_offset + height - (stroke_width / 2.0)]);

    var x = d3.scaleLinear()
        .domain([min, max])
        .range([x_offset + (stroke_width / 2.0), x_offset + width - (stroke_width / 2.0)]);

    if (debug) {
        svg.append("text")
            .attr("class", "timestep-" + timestep)
            .attr("x", x_offset)
            .attr("y", y_offset - 2)
            .style("font-size", "12px")
            .style("fill", "red")
            .text(name);
    }

    // boundary box
    svg.append("rect")
        .attr("class", "timestep-" + timestep)
        .attr("x", x_offset + 0.5)
        .attr("y", y_offset + 0.5)
        .attr("width", width - 1)
        .attr("height", height - 1)
        .attr("stroke", light_grey)
        .attr("stroke-width", 1)
        .attr("fill", "none");
    svg.append("line")
        .attr("class", "timestep-" + timestep)
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
            .attr("class", "timestep-" + timestep)
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
            // Using the colour 'none' results in the inner portion of the rectange not being clickable.
            // So we fill with the background colour (white) to make it clickable.
            .attr("fill", function(d) { return colour == "none" ? "white" : colour; })
            .on("click", function(d) {
                // TODO
                d3.json("weight-explain?name=" + encodeURI(name) + "&column=" + d.position)
                    .get(function (error, explain_data) { console.log(explain_data); });
            });
    svg.selectAll(".bar")
        .data(vector)
        .enter()
            .append("text")
            .attr("class", "timestep-" + timestep)
            .attr("x", function (d) {
                return x_offset + Math.abs(x(d.value) - x(min)) + 5;
            })
            .attr("y", function (d) {
                return y(d.position) + (y.step() / 2) + 4;
            })
            .style("font-size", "12px")
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
                .style("fill", "red");
        }
    }

    //console.log(line_data);
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

    //console.log(line_data);
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



var total_width = 1200;
var layer_height = 200;
var input_width = 100;
var margin = 25;
var w = 30;
var h = layer_height / 3.0;
var svg = null;

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

function drawLayer(timestep, json) {
    console.log(json);
    $(".timestep-" + timestep).remove();

    var x_offset = (margin * 2) + input_width;
    var y_offset = margin + (timestep * layer_height)
    var operand_height = (h * 2.0 / 5.0);
    var operator_height = (h - (operand_height * 2));

    // gridlines
    var x;
    for (x = 0; x <= total_width; x += w) {
        svg.append("line")
            .attr("class", "timestep-" + timestep)
            .attr("x1", x_offset + x - 0.05)
            .attr("y1", y_offset)
            .attr("x2", x_offset + x - 0.05)
            .attr("y2", y_offset + layer_height)
            .attr("stroke-dasharray", "5,5")
            .attr("stroke", "black")
            .attr("stroke-width", 0.1);
    }
    var y;
    for (y = 0; y <= layer_height; y += (h / 2.0)) {
        svg.append("line")
            .attr("class", "timestep-" + timestep)
            .attr("x1", x_offset)
            .attr("y1", y_offset + y - 0.05)
            .attr("x2", x_offset + total_width)
            .attr("y2", y_offset + y - 0.05)
            .attr("stroke-dasharray", "5,5")
            .attr("stroke", "black")
            .attr("stroke-width", 0.1);
    }

    //draw embedding
    drawWeightVector(timestep, json.embedding, "embedding",
        x_offset + (w), y_offset + (h / 2),
        w, h);

    // Draw units
    var u;
    for (u = 0; u < json.units.length; u++) {
        var unit_offset = u * w * 16;
        drawWeightVector(timestep, json.units[u].cell_previous, "cell_previous-" + u,
            x_offset + (w * 3) + (w / 2) + unit_offset, y_offset,
            w, operand_height);
        drawWeightVector(timestep, json.units[u].forget_gate, "forget_gate-" + u,
            x_offset + (w * 3) + (w / 2) + unit_offset, y_offset + (h * 1 / 2) + (operator_height / 2),
            w, operand_height);
        drawWeightVector(timestep, json.units[u].forget, "forget-" + u,
            x_offset + (w * 5) + unit_offset, y_offset,
            w, h);
        drawWeightVector(timestep, json.units[u].input_hat, "input_hat-" + u,
            x_offset + (w * 7) + (w / 2) + unit_offset, y_offset + h,
            w, operand_height);
        drawWeightVector(timestep, json.units[u].remember_gate, "remember_gate-" + u,
            x_offset + (w * 7) + (w / 2) + unit_offset, y_offset + (h * 3 / 2) + (operator_height / 2),
            w, operand_height);
        drawWeightVector(timestep, json.units[u].remember, "remember-" + u,
            x_offset + (w * 9) + unit_offset, y_offset + h,
            w, h);
        drawWeightVector(timestep, json.units[u].forget, "forget-" + u,
            x_offset + (w * 11) + (w / 2) + unit_offset, y_offset + (h * 1 / 2),
            w, operand_height);
        drawWeightVector(timestep, json.units[u].remember, "remember-" + u,
            x_offset + (w * 11) + (w / 2) + unit_offset, y_offset + (h * 2 / 2) + (operator_height / 2),
            w, operand_height);
        drawWeightVector(timestep, json.units[u].cell, "cell-" + u,
            x_offset + (w * 13) + unit_offset, y_offset + (h * 1 / 2),
            w, h);
        drawWeightVector(timestep, json.units[u].cell_hat, "cell_hat-" + u,
            x_offset + (w * 15) + (w / 2) + unit_offset, y_offset + (h / 2),
            w, operand_height);
        drawWeightVector(timestep, json.units[u].output_gate, "output_gate-" + u,
            x_offset + (w * 15) + (w / 2) + unit_offset, y_offset + h + (operator_height / 2),
            w, operand_height);
        drawWeightVector(timestep, json.units[u].output, "output-" + u,
            x_offset + (w * 17) + unit_offset, y_offset + (h / 2),
            w, h);
    }

    // Draw softmax
    drawLabelWeightVector(timestep, json.softmax, "softmax", x_offset + (json.units.length * w * 17) + (w * 3 / 2), y_offset + (h / 2), w, h);

    //draw signs
    drawOperationSign("dotProduct", 400, 250, 60, '#abc');
    drawOperationSign("addition", 400, 350, 60, '#abc');
    drawOperationSign("equals", 400, 450, 60, '#abc');
}

function drawWeightVector(timestep, weight, name, x_offset, y_offset, width, height) {
    drawWeightWidget(x_offset, y_offset, width, height, weight.minimum, weight.maximum, weight.vector, weight.colour, name, timestep);
}

function drawLabelWeightVector(timestep, label_weight, name, x_offset, y_offset, width, height) {
    drawWeightWidget(x_offset, y_offset, width, height, label_weight.minimum, label_weight.maximum, label_weight.vector, "none", name, timestep);
}

function drawWeightWidget(x_offset, y_offset, width, height, min, max, vector, colour, name, timestep) {
    var strokeWidth = 2;

    var y = d3.scaleBand()
        .domain(vector.map(function (d) { return d.position; }))
        .range([y_offset + (strokeWidth / 2.0), y_offset + height - (strokeWidth / 2.0)]);

    var x = d3.scaleLinear()
        .domain([min, max])
        .range([(x_offset + strokeWidth / 2.0), x_offset + width - (strokeWidth / 2.0)]);

    svg.append("text")
        .attr("class", "timestep-" + timestep)
        .attr("x", x_offset)
        .attr("y", y_offset - 2)
        .style("font-size", "12px")
        .text(name);
    // boundary box
    svg.append("rect")
        .attr("class", "timestep-" + timestep)
        .attr("x", x_offset + 0.5)
        .attr("y", y_offset + 0.5)
        .attr("width", width - 1)
        .attr("height", height - 1)
        .attr("stroke", "black")
        .attr("stroke-width", 1)
        .attr("fill", "none");
    svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x(0))
        .attr("y1", y.range()[0])
        .attr("x2", x(0))
        .attr("y2", y.range()[1])
        .attr("stroke", "black")
        .attr("stroke-width", strokeWidth);
    // append the rectangles for the bar chart
    svg.selectAll(".bar")
        .data(vector)
        .enter().append("rect")
        .attr("class", "timestep-" + timestep)
        .attr("x", function (d) {
            return x(Math.min(0, d.value));
        })
        .attr("y", function (d) {
            return y(d.position);
        })
        .attr("width", function (d) {
            return Math.abs(x(d.value) - x(0));
        })
        .attr("height", y.bandwidth())
        .attr("stroke", "black")
        .attr("stroke-width", strokeWidth)
        .attr("fill", colour);
}

function drawOperationSign(operation, x_offset, y_offset, size, colour) {
    var strokeWidth = 2;

    var operationData = [{
        "center": 0.5 * size,
        "elementRadius": 0.4 * size,

        "dotProductRadius": 0.125 * size,

        "plusRectLongSide": 0.6 * size,
        "plusRectShortSide": 0.08 * size,

        "equalsRectLongSide": 0.5 * size,
        "equalsRectShortSide": 0.08 * size
    }];

    var svgContainer = d3.select("body").append("svg")
        .style('position', 'absolute')
        .style('top', x_offset)
        .style('left', y_offset)
        .style('width', size)
        .style('height', size);

    var operator = svgContainer.selectAll("g")
        .data(operationData)
        .enter()
        .append("g");
    operator //background
        .append("circle")
        .attr("cx", function (d) { return d.center; })
        .attr("cy", function (d) { return d.center; })
        .attr("r", function (d) { return d.elementRadius; })
        .style("fill", function () { return colour; })
        .attr("stroke-width", strokeWidth)
        .attr("stroke", "black");


    switch (operation) {
        case "dotProduct":
            operator //dot product
                .append("circle")
                .attr("cx", function (d) { return d.center; })
                .attr("cy", function (d) { return d.center; })
                .attr("r", function (d) { return d.dotProductRadius; })
                .style("fill", function () { return "black"; });
            break;
        case "addition":
            operator //horizontal plus rect
                .append("rect")
                .attr("x", function (d) { return d.center - d.plusRectShortSide * 3.75; })
                .attr("y", function (d) { return d.center - d.plusRectShortSide * 0.5; })
                .attr("width", function (d) { return d.plusRectLongSide; })
                .attr("height", function (d) { return d.plusRectShortSide; })
                .style("fill", function () { return "black"; });
            operator //vertical plus rect
                .append("rect")
                .attr("y", function (d) { return d.center - d.plusRectShortSide * 3.75; })
                .attr("x", function (d) { return d.center - d.plusRectShortSide * 0.5; })
                .attr("height", function (d) { return d.plusRectLongSide; })
                .attr("width", function (d) { return d.plusRectShortSide; })
                .style("fill", function () { return "black"; });
            break;
        case "equals":
            operator //equals upper rect
                .append("rect")
                .attr("x", function (d) { return d.center - d.equalsRectShortSide * 3.2; })
                .attr("y", function (d) { return d.center - d.equalsRectShortSide * 1.5; })
                .attr("width", function (d) { return d.equalsRectLongSide; })
                .attr("height", function (d) { return d.equalsRectShortSide; })
                .style("fill", function () { return "black"; });
            operator //equals lower rect
                .append("rect")
                .attr("x", function (d) { return d.center - d.equalsRectShortSide * 3.2; })
                .attr("y", function (d) { return d.center + d.equalsRectShortSide; })
                .attr("width", function (d) { return d.equalsRectLongSide; })
                .attr("height", function (d) { return d.equalsRectShortSide; })
                .style("fill", function () { return "black"; });
            break;
        default:
            break;
    }
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

        var i;
        for (i = 0; i < words.length; i++) {
            if (words[i].substr(0, value.length).toUpperCase() === value.toUpperCase()) {
                autocomplete.append("<div class='autocomplete-option'>" + words[i] + "</div>");
            }
        }

        $(".autocomplete-option").on("click", function(e) {
            autocomplete.find(".autocomplete-option").remove();
            autocomplete.find("input").val(e.target.textContent);
            console.log("neural-network?sequence=" + encodeURI(e.target.textContent));
            d3.json("neural-network?sequence=" + encodeURI(e.target.textContent))
                .get(function (error, data) { drawLayer(timestep, data); });
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


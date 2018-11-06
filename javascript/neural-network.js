
var total_width = 1000;
var layer_height = 200;
var svg = null;

$(document).ready(function() {
    svg = d3.select('body').append('svg')
        .style('position', 'absolute')
        .style('top', 0)
        .style('left', 0)
        .style('width', total_width * 2)
        .style('height', layer_height * 5);

    d3.json("neural-network")
        .get(function(error, data) { drawLayer(0, data); });
});

function drawLayer(layer, json) {
    console.log(json);

    var x_offset = 50;
    var y_offset = 50 + (layer * layer_height)
    var w = 30;
    var h = layer_height / 3.0;
    var operand_height = (h * 2.0 / 5.0);
    var operator_height = (h - (operand_height * 2));

    // gridlines
    var x;
    for (x = 0; x <= total_width; x += w) {
        svg.append("line")
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
            .attr("x1", x_offset)
            .attr("y1", y_offset + y - 0.05)
            .attr("x2", x_offset + total_width)
            .attr("y2", y_offset + y - 0.05)
            .attr("stroke-dasharray", "5,5")
            .attr("stroke", "black")
            .attr("stroke-width", 0.1);
    }

    //draw embedding
    drawWeightVector(json.embedding, "embedding",
        x_offset + (w), y_offset,
        w, h);

    //draw first vector
    drawWeightVector(json.units[0].c_previous, "c_previous",
        x_offset + (w * 3) + (w / 2), y_offset + (h * 1 / 2),
        w, operand_height);
    drawWeightVector(json.units[0].forget_gate, "forget_gate",
        x_offset + (w * 3) + (w / 2), y_offset + (h * 2 / 2) + (operator_height / 2),
        w, operand_height);
    drawWeightVector(json.units[0].c_hat, "todo1",
        x_offset + (w * 5), y_offset + (h * 1 / 2),
        w, h);
    drawWeightVector(json.units[0].input_hat, "input_hat",
        x_offset + (w * 7) + (w / 2), y_offset + (h * 3 / 2),
        w, operand_height);
    drawWeightVector(json.units[0].remember_gate, "remember_gate",
        x_offset + (w * 7) + (w / 2), y_offset + (h * 4 / 2) + (operator_height / 2),
        w, operand_height);
    drawWeightVector(json.units[0].c_hat, "todo2",
        x_offset + (w * 9), y_offset + (h * 3 / 2),
        w, h);
    drawWeightVector(json.units[0].c_hat, "todo1'",
        x_offset + (w * 11) + (w / 2), y_offset + (h * 1 / 2),
        w, operand_height);
    drawWeightVector(json.units[0].c_hat, "todo2'",
        x_offset + (w * 11) + (w / 2), y_offset + (h * 2 / 2) + (operator_height / 2),
        w, operand_height);
    drawWeightVector(json.units[0].c, "c",
        x_offset + (w * 13), y_offset + (h * 1 / 2),
        w, h);
    drawWeightVector(json.units[0].c_hat, "c_hat",
        x_offset + (w * 15) + (w / 2), y_offset + (h * 1 / 2) + (operator_height / 2),
        w, operand_height);
    drawWeightVector(json.units[0].output_gate, "output_gate",
        x_offset + (w * 15) + (w / 2), y_offset,
        w, operand_height);
    drawWeightVector(json.units[0].output, "output",
        x_offset + (w * 17), y_offset,
        w, h);

    //draw softmax
    drawLabelWeightVector(json.softmax, "softmax", x_offset + (w * 20), y_offset, w, h);

    //draw signs
    drawOperationSign("dotProduct", 400, 250, 60, '#abc');
    drawOperationSign("addition", 400, 350, 60, '#abc');
    drawOperationSign("equals", 400, 450, 60, '#abc');
}

function drawWeightVector(weight, name, x_offset, y_offset, width, height) {
    drawWeightWidget(x_offset, y_offset, width, height, weight.minimum, weight.maximum, weight.vector, weight.colour, name);
}

function drawLabelWeightVector(label_weight, name, x_offset, y_offset, width, height) {
    drawWeightWidget(x_offset, y_offset, width, height, label_weight.minimum, label_weight.maximum, label_weight.vector, "none", name);
}

function drawWeightWidget(x_offset, y_offset, width, height, min, max, vector, colour, name) {
    var strokeWidth = 2;

    var y = d3.scaleBand()
        .domain(vector.map(function (d) { return d.position; }))
        .range([y_offset + (strokeWidth / 2.0), y_offset + height - (strokeWidth / 2.0)]);

    var x = d3.scaleLinear()
        .domain([min, max])
        .range([(x_offset + strokeWidth / 2.0), x_offset + width - (strokeWidth / 2.0)]);

    svg.append("text")
        .attr("x", x_offset)
        .attr("y", y_offset - 2)
        .style("font-size", "12px")
        .text(name);
    // boundary box
    svg.append("rect")
        .attr("x", x_offset + 0.5)
        .attr("y", y_offset + 0.5)
        .attr("width", width - 1)
        .attr("height", height - 1)
        .attr("stroke", "black")
        .attr("stroke-width", 1)
        .attr("fill", "none");
    svg.append("line")
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



$(document).ready(function() {

    d3.json("neural-network")
        .get(function(error, data) { draw(data); });
});

function draw(json) {
    console.log(json);

    var w = 50;
    var h = 50;

    //draw embedding
    drawWeightVector(30, 30, 100, 100, json.embedding);

    //draw first vector
    drawWeightVector(30, 140, w, h, json.units[0].c);
    drawWeightVector(30, 200, w, h, json.units[0].c_hat);
    drawWeightVector(30, 260, w, h, json.units[0].c_previous);
    drawWeightVector(30, 320, w, h, json.units[0].forget_gate);
    drawWeightVector(30, 380, w, h, json.units[0].input_hat);
    drawWeightVector(30, 440, w, h, json.units[0].remember_gate);
    drawWeightVector(30, 500, w, h, json.units[0].output_gate);
    drawWeightVector(30, 560, w, h, json.units[0].output);

    //draw softmax
    drawLabelWeightVector(30, 620, 100, 100, json.softmax);

    //draw signs
    drawOperationSign("dotProduct", 200, 250, 60, '#abc');
    drawOperationSign("addition", 200, 350, 60, '#abc');
    drawOperationSign("equals", 200, 450, 60, '#abc');
}

function drawWeightVector(top, left, width, height, weight) {
    drawWeightWidget(top, left, width, height, weight.minimum, weight.maximum, weight.vector, weight.colour);
}

function drawLabelWeightVector(top, left, width, height, labelWeight) {
    drawWeightWidget(top, left, width, height, 0, 1, labelWeight.vector, "none");
}

function drawWeightWidget(top, left, width, height, min, max, vector, colour) {
    var strokeWidth = 2;

    // Add svg to
    var svg = d3.select('body').append('svg')
        .style('position', 'absolute')
        .style('top', top)
        .style('left', left)
        .style('width', width)
        .style('height', height)
        .append('g');

    var y = d3.scaleBand()
        .domain(vector.map(function (d) { return d.position; }))
        .range([(strokeWidth / 2.0), height - (strokeWidth / 2.0)]);

    var x = d3.scaleLinear()
        .domain([min, max])
        .range([(strokeWidth / 2.0), width - (strokeWidth / 2.0)]);

    // boundary box
    svg.append("rect")
        .attr("x", 0.5)
        .attr("y", 0.5)
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

function drawOperationSign(operation, top, left, size, colour) {
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
        .style('top', top)
        .style('left', left)
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


$(document).ready(function () {

    var top = 100;
    var left = 100;
    var width = 200;
    var heigth = 200;
    var min = -1;
    var max = 1;
    var vector = [-1, 0.25, -0.5, -0.1, 1];

    drawWeightsForVector(top, left, width, heigth, min, max, vector);

    drawWeightsForVector(top + 30, left + 400, width / 2.0, heigth, min, max, vector);

    drawOperationSign("dotProduct", 320, 100, 80, '#abc');

    drawOperationSign("addition", 320, 400, 40, '#abc');

    drawOperationSign("equals", 320, 700, 200, '#abc');

});

function drawWeightsForVector(top, left, width, height, min, max, vector) {

    data = [];

    for (i = 0; i < vector.length; i++)
        data[i] = { value: vector[i], label: i };

    data = data.reverse();
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
        .domain(data.map(function (d) { return d.label; }))
        .range([height - (strokeWidth / 2.0), strokeWidth / 2.0]);

    var x = d3.scaleLinear()
        .domain([min, max])
        .range([strokeWidth / 2.0, width - (strokeWidth / 2.0)]);

    // boundary box
    svg.append("rect")
        .attr("x", 0.5)
        .attr("y", 0.5)
        .attr("width", width - 1)
        .attr("height", height - 1)
        .attr("stroke", "black")
        .attr("stroke-width", 1)
        .attr("fill", "none");
    // append the rectangles for the bar chart
    svg.selectAll(".bar")
        .data(data)
        .enter().append("rect")
        .attr("x", function (d) {
            return x(Math.min(0, d.value));
        })
        .attr("y", function (d) {
            return y(d.label);
        })
        .attr("width", function (d) {
            return Math.abs(x(d.value) - x(0));
        })
        .attr("height", y.bandwidth())
        .attr("stroke", "black")
        .attr("stroke-width", strokeWidth)
        .attr("fill", "none");
}


function drawOperationSign(operation, top, left, size, color) {

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
        .style("fill", function () { return color; })
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


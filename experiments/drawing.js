$(document).ready(function () {

    var top = 100;
    var left = 100;
    var width = 200;
    var heigth = 200;
    var min = -1;
    var max = 1;
    var vector = [-1, 0.25, -0.5, -0.1, 1];

    drawWeightsForVector(top, left, width, heigth, min, max, vector);

    drawWeightsForVector(top+30, left+400, width/2.0, heigth, min, max, vector);

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

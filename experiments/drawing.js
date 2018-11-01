$(document).ready(function () {

    var top = 100;
    var left = 100;
    var width = 200;
    var heigth = 200;
    var min = -1;
    var max = 1;
    var vector = [0.5, 0.25, -0.5, -0.1, 1];

    drawWeightsForVector(top, left, width, heigth, min, max, vector);

    drawWeightsForVector(top+30, left+400, width, heigth, min, max, vector);

});

function drawWeightsForVector(top, left, width, height, min, max, vector) {

    data = [];

    for (i = 0; i < vector.length; i++)
        data[i] = { value: vector[i], label: i };

    data = data.reverse();

    // Add svg to
    var svg = d3.select('body').append('svg')
        .style('position', 'absolute')
        .style('top', top)
        .style('left', left)
        .style('width', width)
        .style('height', height)
        .style('border', '1px solid black')
        .append('g');

    var y = d3.scaleBand()
        .range([height, 0])
        .padding(0.1);

    var x = d3.scaleLinear()
        .range([0, width]);

    // Scale the range of the data in the domains
    x.domain([min, max]);
    y.domain(data.map(function (d) {
        return d.label;
    }));

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
        .attr("fill", "dimgrey");

    //y-axis

    svg.append("rect")
        .attr("x", width/2-1)
        .attr("y", 0)
        .attr("width", 2)
        .attr("height", height)
        .attr("fill", "black");
}
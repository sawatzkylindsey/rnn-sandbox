
$(document).ready(function() {

    d3.json("neural-network")
        .get(function(error, data) { draw(data); });
});

function draw(json) {
    console.log(json);

    var min = -1;
    var max = 1;
    var w = 50;
    var h = 50;

    //draw embedding
    drawWeightsForVector(30, 30, 100, 100, min, max, json.embedding.vector, json.embedding.colour);

    //draw first vector
    
    drawWeightsForVector(30, 140, w, h, min, max, json.layers[0].c.vector, json.layers[0].c.colour);
    drawWeightsForVector(30, 200, w, h, min, max, json.layers[0].c_hat.vector, json.layers[0].c_hat.colour);
    drawWeightsForVector(30, 260, w, h, min, max, json.layers[0].c_previous.vector, json.layers[0].c_previous.colour);
    drawWeightsForVector(30, 320, w, h, min, max, json.layers[0].forget_gate.vector, json.layers[0].forget_gate.colour);
    drawWeightsForVector(30, 380, w, h, min, max, json.layers[0].input_hat.vector, json.layers[0].input_hat.colour);
    drawWeightsForVector(30, 440, w, h, min, max, json.layers[0].output.vector, json.layers[0].output.colour);
    drawWeightsForVector(30, 500, w, h, min, max, json.layers[0].output_gate.vector, json.layers[0].output_gate.colour);
    drawWeightsForVector(30, 560, w, h, min, max, json.layers[0].output.vector, json.layers[0].output.colour);

    //draw output
    drawWeightsForVector(30, 620, 100, 100, min, max, json.output.vector, json.output.colour);
}

function drawWeightsForVector(top, left, width, height, min, max, vector, color) {

    data = [];

    for (i = 0; i < vector.length; i++) {

        data[i] = { value: vector[i].value, label: vector[i].position };
    }

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
        .range([height - strokeWidth / 2.0, strokeWidth / 2.0]);

    var x = d3.scaleLinear()
        .domain([min, max])
        .range([strokeWidth / 2.0, width - strokeWidth / 2.0]);

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
            return d.value===0 ? 0.000001 : Math.abs(x(d.value) - x(0));
        })
        .attr("height", y.bandwidth())
        .attr("stroke", "black")
        .attr("stroke-width", strokeWidth)
        .attr("fill", color);
}



var WIDTH = 1000;
var HEIGHT = 600;
var svg = null;

$(document).ready(function() {
    svg = d3.select("svg");
    svg.attr("width", WIDTH)
        .attr("height", HEIGHT);
    svg.append("circle")
        .style("stroke-width", 5)
        .style("stroke", "blue")
        .style("fill", "white")
        .attr("r", 20)
        .attr("cx", 0)
        .attr("cy", 0)
        .attr("transform", "translate(100,100)");

    d3.json("neural-network")
        .get(function(error, data) { draw(data); });
});

function draw(layer) {
    console.log(layer);
    svg.selectAll("g").remove();
    /*var y = d3.scaleLinear()
        .domain([0, 1])
        .range([0, 50]);
    var y = d3.scaleBand()
        .range([startY, startY + (barNodes.length * barHeight)])
        .domain(barNodes.map(function(d) { return d.name; }));*/
    svg.append("g")
        .selectAll("whatever")
        .data(layer.embedding.vector)
        .enter()
            .append("rect")
            .attr("x", 10)
            .attr("y", function(d) { return 10 + (d.position * 10); })
            .attr("width", function(d) { return 20 * d.value; })
            .attr("height", 10)
            .attr("fill", layer.embedding.colour);

}


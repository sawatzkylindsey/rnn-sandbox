$(document).ready(function () {

    var x = 30, y = 30, width = 50, height = 100, vector = 0;

    //drawWeight(x, y, width, height, vector);
    testHorizontalBars();

});


function drawWeight(x, y, width, height, vector) {
    var svgContainer = d3.select("body").append("svg")
        .attr("width", 300)
        .attr("height", 300);

    //Draw the weight
    var rectangle = svgContainer.append("rect")
        .attr("x", x)
        .attr("y", y)
        .attr("width", width)
        .attr("height", height)
        .attr("fill", "red");
}

function testHorizontalBars() {
    var margin = { top: 30, right: 10, bottom: 50, left: 50 },
        width = 500,
        height = 300;

    var data = [{ value: -10, label: "element 1" },
    { value: 40, label: "element 2" },
    { value: -10, label: "element 3" },
    { value: -50, label: "element 4" },
    { value: 30, label: "element 5" },
    { value: -20, label: "element 6" },
    { value: -70, label: "element 7" }];

    data = data.reverse();

    // Add svg to
    var svg = d3.select('body').append('svg').attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    var y = d3.scaleBand()
        .range([height, 0])
        .padding(0.1);

    var x = d3.scaleLinear()
        .range([0, width]);

    // Scale the range of the data in the domains
    x.domain(d3.extent(data, function (d) {
        return d.value;
    }));
    y.domain(data.map(function (d) {
        return d.label;
    }));

    // append the rectangles for the bar chart
    svg.selectAll(".bar")
        .data(data)
        .enter().append("rect")
        .attr("class", function (d) {
            return "bar bar--" + (d.value < 0 ? "negative" : "positive");
        })
        .attr("x", function (d) {
            return x(Math.min(0, d.value));
        })
        .attr("y", function (d) {
            return y(d.label);
        })
        .attr("width", function (d) {
            return Math.abs(x(d.value) - x(0));
        })
        .attr("height", y.bandwidth());

    // add the x Axis
    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x));

    // add the y Axis
    let yAxisGroup = svg.append("g")
        .attr("class", "y axis")
        .attr("transform", "translate(" + x(0) + ",0)")
        .call(d3.axisRight(y));
    yAxisGroup.selectAll('.tick')
        .data(data)
        .select('text')
        .attr('x', function (d, i) { return d.value < 0 ? 9 : -9; })
        .style('text-anchor', function (d, i) { return d.value < 0 ? 'start' : 'end'; });
}
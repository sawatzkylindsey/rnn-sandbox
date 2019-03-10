
var MEMORY_CHIP_HEIGHT = 5;
var MEMORY_CHIP_WIDTH = 3;
var DETAIL_CHIP_WIDTH = 5;
var total_width = null;
var total_height = null;
var detail_margin = 10;
var layer_height = 225;
var input_width = 100;
var x_margin = 20;
var y_margin = 25;
var HEIGHT = 20;
var circle_radius = 8;
var state_width = 30;
var state_height = layer_height / 3.0;
state_height = 50;
if ((state_height / MEMORY_CHIP_HEIGHT) != (state_width / MEMORY_CHIP_WIDTH)) {
    throw "chips aren't square (" + (state_width / MEMORY_CHIP_WIDTH) + ", " + (state_height / MEMORY_CHIP_HEIGHT) + ")";
}
var operand_height = (state_height * 2.0 / 5.0);
var operator_height = (state_height - (operand_height * 2));
var black = "#3f3f3f";
var dark_grey = "#7e7e7e";
var light_grey = "#bdbdbd";
var dark_red = "#e60000";
var light_red = "#ff1919";
var dark_blue = "#0000e6";
var hash_parts = window.location.hash.substring(1).split(",");
var debug = hash_parts.indexOf("debug") >= 0;
var svg = null;
var main_sequence = [];
var main_timestep = null;
var compare_sequence = [];
var compare_timestep = null;
var input_part = null;
var input_layer = null;
var words = null;
var qb_width = 200;
var qb_height = 600;
var predicate_margin = 10;
var predicate_width = qb_width - (predicate_margin * 2);
var predicate_height = 40;
var queryBuilder = null;
var query = [];
var query_timestep = null;
var maximum_query_length = 6;
var estimate_latest = null;
var active_components = {
    "embedding_hidden": true,
    "input_hat": false,
    "forget": false,
    "remember": false,
    "cell": false,
    "output": true,
    "softmax": true,
};
var notationOff = 0.5;

$(document).ready(function () {
    total_width = TOTAL_WIDTH - LEFT_WIDTH;
    total_height = TOTAL_HEIGHT - TOP_HEIGHT;
    d3.json("words")
        .get(function (error, data) { words = data; });

    svg = d3.select('body').append('svg')
        .attr("transform", "translate(" + LEFT_WIDTH + "," + TOP_HEIGHT + ")")
        .style('position', 'absolute')
        .style("width", total_width)
        .style('height', total_height);
    queryBuilder = d3.select("#query-builder");

    // arrow head definition
    svg.insert('defs', ':first-child')
        .append('marker')
        .attr('id', 'arrow')
        .attr('markerUnits', 'strokeWidth')
        .attr('markerWidth', 12)
        .attr('markerHeight', 12)
        .attr('viewBox', '0 0 12 12')
        .attr('refX', 12)
        .attr('refY', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M2,2 L10,6 L2,10 L6,6 L2,2')
        .style('fill', light_grey);

    function acceptMainInput(sequence) {
        var tail = main_sequence.length;
        main_sequence = sequence;
        trimSequenceTail(tail, main_sequence.length);
        drawWeightsFromSequence(0);
        for (var timestep = 0; timestep < main_sequence.length; timestep++) {
            drawAutocomplete(timestep);
            var autocomplete = $("#autocomplete-" + timestep);
            autocomplete.find("input").val(main_sequence[timestep]);
        }
        drawAutocomplete(timestep);
    }
    $("#header").css("width", TOTAL_WIDTH);
    var main_input = $("#main_input");
    main_input.on("keydown", function(e) {
        // Enter key
        if (e.keyCode == 13) {
            var sequence = $(this).val()
                .toLowerCase()
                .split(/\s+/);
            $(".modal").remove();
            acceptMainInput(sequence);
        }
    });
    $("#query").css("cursor", "pointer")
        .on("click", function(d) {
            swapTab($(this), $("#notation"));
            $("#query-content").css("display", "block");
            $("#notation-content").css("display", "none");
        });
    $("#notation")
        .on("click", function(d) {
            swapTab($(this), $("#query"));
            $("#query-content").css("display", "none");
            $("#notation-content").css("display", "block");
        });
    componentSwitch("embedding_hidden");
    componentSwitch("input_hat");
    componentSwitch("forget");
    componentSwitch("remember");
    componentSwitch("cell");
    componentSwitch("output");
    queryBuilderControls();
    addPredicateTemplate();
});

function componentSwitch(name) {
    var flag = active_components[name];
    $(".notation-" + name)
        .on("mouseover", function(d) {
            if (flag) {
                $(".notation-" + name).css("opacity", notationOff);
            } else {
                $(".notation-" + name).css("opacity", 1);
            }
            $(".notation-" + name).css("cursor", "pointer");
        })
        .on("mouseout", function(d) {
            if (flag) {
                $(".notation-" + name).css("opacity", 1);
            } else {
                $(".notation-" + name).css("opacity", notationOff);
            }
            $(".notation-" + name).css("cursor", "auto");
        })
        .on("click", function(d) {
            flag = !flag;
            active_components[name] = flag;

            if (flag) {
                d3.selectAll(".notation-" + name).style("opacity", 1.0);
                d3.selectAll("." + name).style("opacity", 1);
                d3.selectAll("image." + name).style("opacity", notationOff);
                d3.selectAll(".subtle." + name).style("opacity", 0.2);
            } else {
                d3.selectAll(".notation-" + name).style("opacity", notationOff);
                d3.selectAll("." + name).style("opacity", 0);
            }
        });
}

function queryBuilderControls() {
    queryBuilder.append("text")
        .attr("x", (qb_width / 2) - textWidth("execute", 14) - 10)
        .attr("y", predicate_margin + 14)
        .style("font-size", "14px")
        .style("fill", black)
        .text("execute");
    queryBuilder.append("line")
        .attr("x1", (qb_width / 2))
        .attr("y1", predicate_margin)
        .attr("x2", (qb_width / 2))
        .attr("y2", predicate_margin + 20)
        .attr("stroke", dark_grey)
        .attr("stroke-width", 1);
    queryBuilder.append("text")
        .attr("id", "sequence-match-expected")
        .attr("x", (qb_width / 2) + 10)
        .attr("y", predicate_margin + 14)
        .style("font-size", "14px")
        .style("fill", black)
        .text("-");
    queryBuilder.append("rect")
        .attr("id", "execute-box")
        .attr("x", (qb_width / 2) - (textWidth("execute", 14)) - 15)
        .attr("y", predicate_margin)
        .attr("width", textWidth("execute", 14) + textWidth("-", 14) + 30)
        .attr("height", HEIGHT)
        .attr("rx", 5)
        .attr("ry", 5)
        .attr("stroke", dark_grey)
        .attr("stroke-width", 1)
        .attr("fill", "transparent")
        .on("mouseover", function(d) {
            if (query.length > 0) {
                d3.select(this)
                    .style("cursor", "pointer");
                d3.select(this)
                    .transition()
                    .duration(100)
                    .attr("stroke-width", 2);
            }
        })
        .on("mouseout", function(d) {
            if (query.length > 0) {
                d3.select(this)
                    .style("cursor", "auto");
                d3.select(this)
                    .transition()
                    .duration(50)
                    .attr("stroke-width", 1);
            }
        })
        .on("click", function(d) {
            if (query.length > 0) {
                d3.selectAll(".predicate")
                    .style("stroke", dark_grey);
                d3.select(this)
                    .style("cursor", "auto");
                d3.select(this)
                    .transition()
                    .duration(50)
                    .attr("stroke-width", 1);
                closeDetail();
                drawView("Sequence Matches View", "sequence", function() {
                    console.log("Drawing sequence for query:");
                    console.log(query);
                    var params = "";
                    for (var index in hash_parts) {
                        if (hash_parts[index].startsWith("tolerance=")) {
                            params += "&" + hash_parts[index];
                        }
                    }
                    d3.json("sequence-matches?" + query.map(p => "predicate=" + predicateString(p)).join("&") + params)
                        .get(function (error, data) { drawSequences(data); });
                });
            }
        });
}

function predicateString(predicate) {
    var units = [];

    for (var unit in predicate) {
        var values = [];

        for (var feature in predicate[unit]) {
            values.push(feature + ":" + predicate[unit][feature]);
        }

        if (values.length > 0) {
            units.push(unit + "|" + values.join(","));
        }
    }

    return units.join(";");
}

function addPredicateTemplate() {
    if (query.length == maximum_query_length) {
        return;
    }

    var index = query.length;
    var x_offset = predicate_margin;
    var y_offset = 60 + (index * predicate_height) + (index * predicate_margin * 2);
    // Crosshair
    queryBuilder.append("circle")
        .attr("class", "predicate predicate-highlightable predicate-" + index)
        .attr("cx", x_offset + (predicate_width / 8))
        .attr("cy", y_offset + (predicate_height / 2))
        .attr("r", 1)
        .attr("stroke", dark_grey)
        .attr("stroke-width", 1)
        .attr("fill", "none");
    queryBuilder.append("line")
        .attr("class", "predicate predicate-highlightable predicate-" + index)
        .attr("x1", x_offset + (predicate_width / 8))
        .attr("y1", y_offset + (predicate_height / 2) - 8)
        .attr("x2", x_offset + (predicate_width / 8))
        .attr("y2", y_offset + (predicate_height / 2) + 8)
        .attr("stroke", dark_grey)
        .attr("stroke-width", 1);
    queryBuilder.append("line")
        .attr("class", "predicate predicate-highlightable predicate-" + index)
        .attr("x1", x_offset + (predicate_width / 8) - 8)
        .attr("y1", y_offset + (predicate_height / 2))
        .attr("x2", x_offset + (predicate_width / 8) + 8)
        .attr("y2", y_offset + (predicate_height / 2))
        .attr("stroke", dark_grey)
        .attr("stroke-width", 1);
    queryBuilder.append("circle")
        .attr("class", "predicate predicate-" + index)
        .attr("cx", x_offset + (predicate_width / 8))
        .attr("cy", y_offset + (predicate_height / 2))
        .attr("r", 8)
        .attr("stroke", "none")
        .attr("fill", "transparent")
        .on("mouseover", function(d) {
            if (input_part != null) {
                d3.select(this)
                    .style("cursor", "pointer");
                d3.selectAll(".predicate-" + index)
                    .transition()
                    .duration(100)
                    .attr("stroke-width", 2);
            }
        })
        .on("mouseout", function(d) {
            if (input_part != null) {
                d3.select(this)
                    .style("cursor", "auto");
                d3.selectAll(".predicate-" + index)
                    .transition()
                    .duration(50)
                    .attr("stroke-width", 1);
            }
        })
        .on("click", function(d) {
            if (input_part != null) {
                query_timestep = index;
                d3.selectAll(".predicate-highlightable")
                    .attr("stroke", dark_grey);
                d3.selectAll(".activation-box")
                    .attr("stroke", light_grey);
                d3.selectAll(".predicate-highlightable.predicate-" + query_timestep)
                    .attr("stroke", dark_blue);
            }
        });
    // Select all
    /*queryBuilder.append("line")
        .attr("class", "predicate predicate-highlightable predicate-" + index)
        .attr("x1", x_offset + (predicate_width / 8) - 8)
        .attr("y1", y_offset + (predicate_height / 2))
        .attr("x2", x_offset + (predicate_width / 8) + 8)
        .attr("y2", y_offset + (predicate_height / 2))
        .attr("stroke", dark_grey)
        .attr("stroke-width", 1);
    queryBuilder.append("circle")
        .attr("class", "predicate predicate-" + index)
        .attr("cx", x_offset + (predicate_width / 8))
        .attr("cy", y_offset + (predicate_height / 2))
        .attr("r", 8)
        .attr("stroke", "none")
        .attr("fill", "transparent")
        .on("mouseover", function(d) {
            if (input_part != null) {
                d3.select(this)
                    .style("cursor", "pointer");
                d3.selectAll(".predicate-" + index)
                    .transition()
                    .duration(100)
                    .attr("stroke-width", 2);
            }
        })
        .on("mouseout", function(d) {
            if (input_part != null) {
                d3.select(this)
                    .style("cursor", "auto");
                d3.selectAll(".predicate-" + index)
                    .transition()
                    .duration(50)
                    .attr("stroke-width", 1);
            }
        })
        .on("click", function(d) {
            if (input_part != null) {
                query_timestep = index;
                d3.selectAll(".predicate-highlightable")
                    .attr("stroke", dark_grey);
                d3.selectAll(".activation-box")
                    .attr("stroke", light_grey);
                d3.selectAll(".predicate-highlightable.predicate-" + query_timestep)
                    .attr("stroke", dark_blue);
            }
        });*/
    // Separator line
    queryBuilder.append("line")
        .attr("class", "predicate predicate-" + index)
        .attr("x1", x_offset + (predicate_width / 2))
        .attr("y1", y_offset)
        .attr("x2", x_offset + (predicate_width / 2))
        .attr("y2", y_offset + predicate_height)
        .attr("stroke", dark_grey)
        .attr("stroke-width", 1);
    queryBuilder.append("rect")
        .attr("class", "predicate predicate-highlightable predicate-" + index)
        .attr("x", x_offset)
        .attr("y", y_offset)
        .attr("width", predicate_width)
        .attr("height", predicate_height)
        .attr("rx", 5)
        .attr("ry", 5)
        .attr("stroke", dark_grey)
        .attr("stroke-width", 1)
        .attr("fill", "none");
    updatePredicateCounts(index);
    queryBuilder.append("line")
        .attr("class", "predicate predicate-" + index)
        .attr("x1", x_offset + (predicate_width / 2) + (predicate_width / 4))
        .attr("y1", y_offset + 10)
        .attr("x2", x_offset + (predicate_width / 2) + (predicate_width / 4))
        .attr("y2", y_offset + predicate_height - 10)
        .attr("stroke", black)
        .attr("stroke-width", 1);
}

function updatePredicateCounts(query_timestep) {
    var units = 0;
    var activations = 0;

    if (query.length > 0) {
        for (var unit in query[query_timestep]) {
            units += 1;

            for (var feature in query[query_timestep][unit]) {
                activations += 1;
            }
        }
    }

    $("#predicate-units-" + query_timestep).remove();
    var x_offset = predicate_margin;
    var y_offset = 60 + (query_timestep * predicate_height) + (query_timestep * predicate_margin * 2);
    queryBuilder.append("text")
        .attr("id", "predicate-units-" + query_timestep)
        .attr("class", "predicate predicate-" + query_timestep)
        .attr("x", x_offset + (predicate_width / 2) + (predicate_width * 1 / 8) - (textWidth(units, 14) / 2))
        .attr("y", y_offset + (predicate_height / 2) + 5)
        .style("font-size", "14px")
        .style("fill", black)
        .text(units);
    $("#predicate-activations-" + query_timestep).remove();
    queryBuilder.append("text")
        .attr("id", "predicate-activations-" + query_timestep)
        .attr("class", "predicate predicate-" + query_timestep)
        .attr("x", x_offset + (predicate_width / 2) + (predicate_width * 3 / 8) - (textWidth(activations, 14) / 2))
        .attr("y", y_offset + (predicate_height / 2) + 5)
        .style("font-size", "14px")
        .style("fill", black)
        .text(activations);
}

function updateSequenceMatchesEstimate() {
    var params = "";

    for (var index in hash_parts) {
        if (hash_parts[index].startsWith("tolerance=")) {
            params += "&" + hash_parts[index];
        }
        if (hash_parts[index] == "exact") {
            params += "&exact=True";
        }
    }

    d3.select("#sequence-match-expected").text("...");
    d3.select("#execute-box").attr("width", textWidth("execute", 14) + textWidth("...", 14) + 30)
    var predicates_qp = query.map(p => "predicate=" + predicateString(p)).join("&");

    if (predicates_qp.length > "predicate=".length * query.length) {
        var now = new Date();
        estimate_latest = now;
        d3.json("sequence-matches-estimate?" + predicates_qp + params)
            .get(function (error, data) {
                if (now < estimate_latest) {
                    return;
                }

                var estimate = "";

                if (data.exact != null) {
                    estimate += data.exact;
                } else {
                    if (data.lower != null) {
                        estimate += data.lower;
                    }

                    if (data.lower != null && data.upper != null) {
                        estimate += " - " + data.upper;
                    } else if (data.upper != null) {
                        estimate += "? - " + data.upper;
                    } else {
                        estimate += " - ?";
                    }
                }

                d3.select("#sequence-match-expected").text(estimate);
                d3.select("#execute-box").attr("width", textWidth("execute", 14) + textWidth(estimate, 14) + 30)
            });
    }
}

function drawPredicateChip(query_timestep, x_offset, y_offset, width, height) {
    var units = 0;
    var activations = 0;

    for (var unit in query[query_timestep]) {
        units += 1;

        for (var feature in query[query_timestep][unit]) {
            activations += 1;
        }
    }

    svg.append("rect")
        .attr("class", "sequence")
        .attr("x", x_offset)
        .attr("y", y_offset)
        .attr("width", width)
        .attr("height", height)
        .attr("rx", 5)
        .attr("ry", 5)
        .attr("stroke", dark_grey)
        .attr("stroke-width", 1)
        .attr("fill", light_grey);
    svg.append("text")
        .attr("class", "sequence")
        .attr("x", x_offset + (width * 1 / 4) - (textWidth(units, 12) / 2))
        .attr("y", y_offset + (height / 2) + 4)
        .style("font-size", "12px")
        .style("fill", black)
        .text(units);
    svg.append("text")
        .attr("class", "sequence")
        .attr("x", x_offset + (width * 3 / 4) - (textWidth(activations, 12) / 2))
        .attr("y", y_offset + (height / 2) + 4)
        .style("font-size", "12px")
        .style("fill", black)
        .text(activations);
    svg.append("line")
        .attr("class", "sequence")
        .attr("x1", x_offset + (width / 2))
        .attr("y1", y_offset + 5)
        .attr("x2", x_offset + (width / 2))
        .attr("y2", y_offset + height - 5)
        .attr("stroke", black)
        .attr("stroke-width", 1);
}

function swapTab(focused, other) {
    focused.css("cursor", "auto")
        .css("border-bottom-style", "none");
    other.css("cursor", "pointer")
        .css("border-bottom-style", "solid");
    $("#lefter").prepend(other);
}

function drawTimestep(fake_timestep, data) {
    console.log("Timestep (fake, actual): (" + fake_timestep + ", " + data.timestep + ")");
    console.log(data);
    $("svg").height(layer_height * (main_sequence.length + 2));
    $(".timestep-" + data.timestep).remove();

    if (data.timestep == 0) {
        drawSubTitle("Component View", "timestep-0 component");
    }

    /*for (var t=0; t < main_sequence.length - 1; t++) {
        $(".timestep-" + t + ".softmax").remove();
    }*/

    if (data.x_word != main_sequence[data.timestep]) {
        svg.append("text")
            .attr("class", "timestep-" + data.timestep + " component")
            .attr("x", x_margin + (input_width * 1 / 3))
            .attr("y", y_margin + (data.timestep * layer_height) + state_height + (state_height / 5) + HEIGHT + 5)
            .style("font-size", "14px")
            .style("fill", black)
            .text(data.x_word);
    }


    var x_offset = (x_margin * 2) + input_width;
    var y_offset = y_margin + (data.timestep * layer_height);
    var operand_height = (state_height * 2.0 / 5.0);
    var operator_height = (state_height - (operand_height * 2));

    if (debug) {
        // gridlines
        for (var x = 0; x <= total_width; x += state_width) {
            svg.append("line")
                .attr("class", "timestep-" + data.timestep + " component")
                .attr("x1", x_offset + x - 0.05)
                .attr("y1", y_offset)
                .attr("x2", x_offset + x - 0.05)
                .attr("y2", y_offset + layer_height)
                .attr("stroke-dasharray", "5,5")
                .attr("stroke", "blue")
                .attr("stroke-width", 0.1);
        }
        for (var y = 0; y <= layer_height; y += (state_height / 2.0)) {
            svg.append("line")
                .attr("class", "timestep-" + data.timestep + " component")
                .attr("x1", x_offset)
                .attr("y1", y_offset + y - 0.05)
                .attr("x2", x_offset + total_width)
                .attr("y2", y_offset + y - 0.05)
                .attr("stroke-dasharray", "5,5")
                .attr("stroke", "blue")
                .attr("stroke-width", 0.1);
        }
    }

    // Draw embedding
    drawHiddenState(data, "embedding");

    // Draw units
    for (var part in data.units) {
        if (data.units.hasOwnProperty(part)) {
            for (var layer in data.units[part]) {
                if (data.units[part].hasOwnProperty(layer)) {
                    drawHiddenState(data, part, layer);
                }
            }
        }
    }

    // Draw softmax
    /*drawHline(timestep, x_offset + (data.units.length * w * 17), y_offset + (h * 2 / 2),
        x_offset + (data.units.length * w * 17) + (w * 3 / 2), y_offset + (h * 2 / 2));*/
    drawSoftmax(data, "softmax");
}

function drawHiddenState(data, part, layer) {
    var geometry = getGeometry(data.timestep, part, layer);

    if (geometry != null) {
        var hiddenState = null;

        if (part in data) {
            hiddenState = data[part];
        } else {
            hiddenState = data.units[part][layer];
        }

        var componentName = nameOf(hiddenState.name);
        var classes = "timestep-" + data.timestep + " component " + componentName;
        drawStateWidget(data.timestep, geometry, hiddenState.name, hiddenState.minimum, hiddenState.maximum, hiddenState.vector, hiddenState.colour, hiddenState.predictions, classes,
            MEMORY_CHIP_WIDTH, MEMORY_CHIP_HEIGHT, part, layer, null, null, null);

        var flag = active_components[componentName];
        if (flag) {
            d3.selectAll("." + componentName).style("opacity", 1);
            d3.selectAll("image." + componentName).style("opacity", notationOff);
            d3.selectAll(".subtle." + componentName).style("opacity", 0.2);
        } else {
            d3.selectAll("." + componentName).style("opacity", 0);
        }
    }
}

function drawStateWidget(timestep, geometry, name, min, max, vector, colour, predictions, classes, chip_width, chip_height, part, layer, linker, linker_suffix, placement) {
    if (min >= max) {
        throw "min " + min + " cannot exceed max " + max;
    }

    if (min > 0) {
        throw "min " + min + " cannot be greater than 0";
    }

    if (max < 0) {
        throw "max " + max + " cannot be less than 0";
    }

    var found_min = d3.min(vector, function(d) { return d.value; });
    if (found_min < min) {
        throw "found value " + found_min + " exceeding min " + min;
    }

    var found_max = d3.max(vector, function(d) { return d.value; });
    if (found_max > max) {
        throw "found value " + found_max + " exceeding max " + max;
    }

    var stroke_width = 1;

    var macro_y = d3.scaleBand()
        .padding(0.2)
        .domain(Array.from(Array(chip_height).keys()))
        .range([geometry.y + (stroke_width / 2.0), geometry.y + geometry.height - (stroke_width / 2.0)]);
    function y(position) {
        return macro_y(position % chip_height);
    }

    var macro_x = d3.scaleBand()
        .padding(0.2)
        .domain(Array.from(Array(chip_width).keys()))
        .range([geometry.x + (stroke_width / 2.0), geometry.x + geometry.width - (stroke_width / 2.0)]);
    function x(position) {
        return macro_x(Math.floor(position / chip_height));
    }

    //var magnitude = d3.scaleLinear()
    //    .domain([1, 1 + Math.max(Math.abs(min), Math.abs(max))])
    var magnitude = d3.scaleLog()
        // In d3 v3 we can't set 0 in the domain, so push everything up by 1.
        // Make sure to do this when applying the scale as well!
        .domain([1, 1 + Math.max(Math.abs(min), Math.abs(max))])
        .range([0, macro_x.bandwidth()]);

    var margin = (geometry.width / 6);

    if (predictions != null) {
        var predictionGeometry = {
            x: geometry.x + geometry.width + (geometry.width / 3) + margin,
            y: geometry.y + (geometry.height / 3),
            width: geometry.width / 3,
            height: geometry.height / 3,
        };
        drawPredictionWidget(timestep, predictionGeometry, null, predictions.minimum, predictions.maximum, predictions.vector, classes, true, name == null ? null : nameOf(name));
        // Draw colour prediction.
        svg.append("rect")
            .attr("class", classes)
            .attr("x", geometry.x + geometry.width + margin - (stroke_width / 2))
            .attr("y", geometry.y + (geometry.height * 1 / 4))
            .attr("width", (geometry.width / 3))
            .attr("height", (geometry.height / 2))
            .attr("stroke", "none")
            .attr("stroke-width", stroke_width)
            .attr("fill", colour == null ? "none" : colour)
            .style("opacity", 1.0);
    }

    if (timestep != null) {
        drawOpen(geometry.x + geometry.width + (margin * 2) - 0.5, geometry.y + margin, margin - 0.5, classes, function () {
            main_timestep = timestep;
            input_part = part;
            input_layer = layer;
            drawView("Detail View", "detail", drawDetail);
        });
    }

    // Name
    if (name != null) {
        svg.append("image")
            .attr("class", classes)
            .attr("xlink:href", "latex/" + name + ".png")
            .attr("x", geometry.x + (geometry.width / 2) - 6)
            .attr("y", geometry.y + geometry.height + 2 + (name.startsWith("e_") ? 5 : 0))
            .style("opacity", notationOff)
            .on("mouseover", function(d) {
                if (active_components[nameOf(name)]) {
                    d3.select(this)
                        .style("opacity", 1.0);
                    var componentName = nameOf(name);

                    if (name.startsWith("c_-")) {
                        componentName = "cell_hidden";
                    }

                    $(".notation-" + componentName).parent()
                        .css("border-color", "black");
                }
            })
            .on("mouseout", function(d) {
                if (active_components[nameOf(name)]) {
                    d3.select(this)
                        .style("opacity", notationOff);
                    var componentName = nameOf(name);

                    if (name.startsWith("c_-")) {
                        componentName = "cell_hidden";
                    }

                    $(".notation-" + componentName).parent()
                        .css("border-color", "transparent");
                }
            });
    }

    // Boundary box
    svg.append("rect")
        .attr("class", classes)
        .attr("x", geometry.x + (stroke_width / 2.0))
        .attr("y", geometry.y + (stroke_width / 2.0))
        .attr("width", geometry.width - stroke_width)
        .attr("height", geometry.height - stroke_width)
        .attr("stroke", light_grey)
        .attr("stroke-width", stroke_width)
        .attr("fill", "none");
    // Chip's colour & magnitude.
    svg.selectAll(".chip")
        .data(vector)
        .enter()
            .append("rect")
            .attr("class", function(d) { return classes + " activation-" + d.position; })
            .attr("data-activation", function(d) { return d.value; })
            .attr("x", function (d) {
                var base_x = x(d.position);

                if (d.value >= 0) {
                    return base_x;
                } else {
                    return base_x + (macro_x.bandwidth() - magnitude(1 + Math.abs(d.value)));
                }
            })
            .attr("y", function (d) {
                if (placement == null || placement == "top") {
                    return y(d.position);
                } else {
                    return y(d.position) + (macro_y.bandwidth() / 2) + 0.5;
                }
            })
            .attr("width", function (d) { return magnitude(1 + Math.abs(d.value)); })
            .attr("height", function (d) {
                if (placement == null) {
                    return macro_y.bandwidth();
                } else {
                    return (macro_y.bandwidth() / 2) - 1;
                }
            })
            .attr("stroke", "none")
            .attr("fill", dark_grey);
    var active_unit = input_part + "," + (input_layer == null ? 0 : input_layer);
    // Chip's scaling box.
    svg.selectAll(".chip")
        .data(vector)
        .enter()
            .append("rect")
            .attr("class", function(d) {
                return classes + " linker-" + (linker == null ? d.position : linker[d.position]) + (linker_suffix == null ? "" : linker_suffix) + " activation-box activation-" + d.position;
            })
            .attr("x", function (d) { return x(d.position); })
            .attr("y", function (d) {
                if (placement == null || placement == "top") {
                    return y(d.position);
                } else {
                    return y(d.position) + (macro_y.bandwidth() / 2) + 0.5;
                }
            })
            .attr("width", macro_x.bandwidth())
            .attr("height", function (d) {
                if (placement == null) {
                    return macro_y.bandwidth();
                } else {
                    return (macro_y.bandwidth() / 2) - 1;
                }
            })
            .attr("stroke", light_grey)
            .attr("stroke-width", stroke_width)
            .attr("fill", "transparent")
            .style("pointer-events", "bounding-box")
            .on("mouseover", function(d) {
                if (timestep == null && $(this).css("opacity") == 1) {
                    if (query_timestep != null) {
                        if (linker != null) {
                            $(this).css("cursor", "crosshair");
                        }
                    } else {
                        d3.selectAll(".linker-" + (linker == null ? d.position : linker[d.position]) + linker_suffix)
                            .transition()
                            .duration(100)
                            .attr("stroke", black)
                            .attr("stroke-width", stroke_width * 2);
                    }
                }
            })
            .on("mouseout", function(d) {
                if (timestep == null && $(this).css("opacity") == 1) {
                    if (query_timestep != null) {
                        if (linker != null) {
                            $(this).css("cursor", "auto");
                        }
                    } else {
                        d3.selectAll(".linker-" + (linker == null ? d.position : linker[d.position]) + linker_suffix)
                            .transition()
                            .duration(50)
                            .attr("stroke", light_grey)
                            .attr("stroke-width", stroke_width);
                    }
                }
            })
            .on("click", function(d) {
                if (timestep == null && $(this).css("opacity") == 1) {
                    if (query_timestep != null && linker != null) {
                        if (query_timestep == query.length) {
                            query.push({});
                        }

                        var predicate = query[query_timestep];

                        if (!(active_unit in predicate)) {
                            predicate[active_unit] = {};
                        }

                        var predicate_unit = predicate[active_unit];

                        if (!(d.position in predicate_unit)) {
                            predicate_unit[d.position] = d.value.toFixed(6);
                            d3.select(this).attr("stroke", dark_blue);
                        } else {
                            delete predicate_unit[d.position];
                            d3.select(this).attr("stroke", light_grey);
                        }

                        updateSequenceMatchesEstimate();
                        updatePredicateCounts(query_timestep);

                        if (query_timestep + 1 == query.length) {
                            addPredicateTemplate();
                        }
                    }
                }
            });
    // Chip's direction line.
    svg.selectAll(".chip")
        .data(vector)
        .enter()
            .append("line")
            .attr("class", function(d) { return classes + " activation-" + d.position; })
            .attr("x1", function(d) {
                var base_x = x(d.position);

                if (d.value >= 0) {
                    return base_x;
                } else {
                    return base_x + macro_x.bandwidth();
                }
            })
            .attr("y1", function (d) {
                if (placement == null || placement == "top") {
                    return y(d.position) - 1;
                } else {
                    return y(d.position) + (macro_y.bandwidth() / 2) - 0.5;
                }
            })
            .attr("x2", function(d) {
                var base_x = x(d.position);

                if (d.value >= 0) {
                    return base_x;
                } else {
                    return base_x + macro_x.bandwidth();
                }
            })
            .attr("y2", function (d) {
                if (placement == null) {
                    return y(d.position) + macro_y.bandwidth() + 1;
                } else if (placement == "bottom") {
                    // Only necessary to manage rounding errors
                    return y(d.position) + macro_y.bandwidth() + 0.5;
                } else if (placement == "top") {
                    return y(d.position) + (macro_y.bandwidth() / 2);
                }
            })
            .attr("stroke", function (d) {
                if (d.value == 0) {
                    return "none";
                }

                return black;
            })
            .attr("stroke-width", stroke_width);
}

function drawSoftmax(data, part) {
    var geometry = getGeometry(data.timestep, part, 1);
    var labelWeightVector = data[part];
    var classes = "timestep-" + data.timestep + " component";
    drawPredictionWidget(data.timestep, geometry, labelWeightVector.name, labelWeightVector.minimum, labelWeightVector.maximum, labelWeightVector.vector, classes, false, null);
}

function drawPredictionWidget(timestep, geometry, name, min, max, predictions, classes, subtle, backComponentName) {
    var found_min = d3.min(predictions, function(d) { return d.value; });
    if (found_min < min) {
        throw "found value " + found_min + " exceeding min " + min;
    }

    var found_max = d3.max(predictions, function(d) { return d.value; });
    if (found_max > max) {
        throw "found value " + found_max + " exceeding max " + max;
    }

    var stroke_width = 1;

    var y = d3.scaleBand()
        .domain(predictions.map(function (d) { return d.position; }))
        .range([geometry.y + (stroke_width / 2.0), geometry.y + geometry.height - (stroke_width / 2.0)]);

    var x = d3.scaleLinear()
        .domain([min, max])
        .range([geometry.x + (stroke_width / 2.0), geometry.x + geometry.width - (stroke_width / 2.0)]);

    var baseOpacity = subtle ? 0.2 : 1.0;
    var id_class = "softmax-" + Math.random().toString(36).substring(5);

    // Name
    if (name != null) {
        svg.append("image")
            .attr("class", classes)
            .attr("xlink:href", "latex/" + name + ".png")
            .attr("x", geometry.x + (geometry.width / 2) - (textWidth(name, 12) / 2))
            .attr("y", geometry.y + geometry.height + 10)
            .style("opacity", notationOff)
            .on("mouseover", function(d) {
                if (active_components[nameOf(name)]) {
                    d3.select(this)
                        .style("opacity", 1.0);
                    $(".notation-" + nameOf(name)).parent()
                        .css("border-color", "black");
                }
            })
            .on("mouseout", function(d) {
                if (active_components[nameOf(name)]) {
                    d3.select(this)
                        .style("opacity", notationOff);
                    $(".notation-" + nameOf(name)).parent()
                        .css("border-color", "transparent");
                }
            });
    }

    svg.append("line")
        .attr("class", classes + (subtle ? " subtle" : " softmax") + " " + id_class)
        .attr("x1", x(0))
        .attr("y1", y.range()[0] - 2)   // Make the center line stand out slightly by pushing it beyond the rectangle.
        .attr("x2", x(0))
        .attr("y2", y.range()[1] + 2)   // Make the center line stand out slightly by pushing it beyond the rectangle.
        .attr("stroke", black)
        .attr("stroke-width", stroke_width)
        .style("opacity", baseOpacity);
    svg.selectAll(".bar")
        .data(predictions)
        .enter()
            .append("rect")
            .attr("class", classes + (subtle ? " subtle" : " softmax") + " " + id_class)
            .attr("x", x(0))
            .attr("y", function (d) {
                return y(d.position);
            })
            // TODO: Why is this more complicated than it needs to be?
            .attr("width", function (d) { return x(d.value) - x(0); })
            .attr("height", y.bandwidth())
            .attr("stroke", black)
            .attr("stroke-width", stroke_width)
            .attr("fill", function(d) {
                if ("colour" in d) {
                    return d.colour == null ? "none" : d.colour;
                }

                return "none";
            })
            .style("opacity", baseOpacity)
            .on("mouseover", function(d) {
                if (backComponentName == null || active_components[backComponentName]) {
                    d3.selectAll("." + id_class)
                        .transition()
                        .duration(100)
                        .style("opacity", 1.0);
                }
            })
            .on("mouseout", function(d) {
                if (backComponentName == null || active_components[backComponentName]) {
                    d3.selectAll("." + id_class)
                        .transition()
                        .duration(50)
                        .style("opacity", baseOpacity);
                }
            });
    svg.selectAll(".bar")
        .data(predictions)
        .enter()
            .append("text")
            .attr("class", classes + (subtle ? " subtle" : " softmax") + " " + id_class)
            .attr("x", function (d) {
                return geometry.x + Math.abs(x(d.value) - x(min)) + 5;
            })
            .attr("y", function (d) {
                return y(d.position) + (y.step() / 2) + (subtle ? 3 : 4);
            })
            .style("font-size", subtle ? "9px" : "12px")
            .style("opacity", baseOpacity)
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
                .style("fill", "blue");
        }
    }

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
            .attr("marker-end", "url(#arrow)")
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
                .style("fill", "blue");
        }
    }

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
            .attr("marker-end", "url(#arrow)")
            .style("fill", "none");
}

function drawOperatorCircle(timestep, x_offset, y_offset, size, addition, parts, class_suffix) {
    svg.append("circle")
        .attr("class", "timestep-" + timestep + " " + class_suffix)
        .attr("cx", x_offset + (size / 2))
        .attr("cy", y_offset + (size / 2))
        .attr("r", (size / 2) - 0.5)
        .attr("stroke", black)
        .attr("stroke-width", 1)
        .attr("fill", light_grey)
        .on("mouseover", function(d) {
            if (parts != null) {
                d3.event.target.style.fill = dark_grey;
            }
        })
        .on("mouseout", function(d) {
            if (parts != null) {
                d3.event.target.style.fill = light_grey;
            }
        })
        .on("click", function(d) {
            if (parts != null) {
                drawZoom(timestep, x_offset + (size / 2), addition, parts);
            }
        });
}

function drawAddition(timestep, x_offset, y_offset, size, parts, class_suffix) {
    drawOperatorCircle(timestep, x_offset, y_offset, size, true, parts, class_suffix);
    var stroke_width = size / 10;
    var stroke_length = size / 2;
    svg.append("line")
        .attr("class", "timestep-" + timestep + " " + class_suffix)
        .attr("x1", x_offset + ((size - stroke_length) / 2))
        .attr("y1", y_offset + (size / 2))
        .attr("x2", x_offset + ((size - stroke_length) / 2) + stroke_length)
        .attr("y2", y_offset + (size / 2))
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
    svg.append("line")
        .attr("class", "timestep-" + timestep + " " + class_suffix)
        .attr("x1", x_offset + (size / 2))
        .attr("y1", y_offset + ((size - stroke_length) / 2))
        .attr("x2", x_offset + (size / 2))
        .attr("y2", y_offset + ((size - stroke_length) / 2) + stroke_length)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
}

function drawMultiplication(timestep, x_offset, y_offset, size, parts, class_suffix) {
    drawOperatorCircle(timestep, x_offset, y_offset, size, false, parts, class_suffix);
    var stroke_width = size / 10;
    var stroke_length = size / 2;
    var qq = Math.sqrt((stroke_length**2) / 2);
    svg.append("line")
        .attr("class", "timestep-" + timestep + " " + class_suffix)
        .attr("x1", x_offset + ((size - qq) / 2))
        .attr("y1", y_offset + ((size - qq) / 2))
        .attr("x2", x_offset + ((size - qq) / 2) + qq)
        .attr("y2", y_offset + ((size - qq) / 2) + qq)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
    svg.append("line")
        .attr("class", "timestep-" + timestep + " " + class_suffix)
        .attr("x1", x_offset + ((size - qq) / 2))
        .attr("y1", y_offset + ((size - qq) / 2) + qq)
        .attr("x2", x_offset + ((size - qq) / 2) + qq)
        .attr("y2", y_offset + ((size - qq) / 2))
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
}

function drawEquals(timestep, x_offset, y_offset, size, class_suffix) {
    drawOperatorCircle(timestep, x_offset, y_offset, size, false, null, class_suffix);
    var stroke_width = size / 10;
    var stroke_length = size / 2;
    svg.append("line")
        .attr("class", "timestep-" + timestep + " " + class_suffix)
        .attr("x1", x_offset + ((size - stroke_length) / 2))
        .attr("y1", y_offset + (size / 2) - (stroke_width * 1))
        .attr("x2", x_offset + ((size - stroke_length) / 2) + stroke_length)
        .attr("y2", y_offset + (size / 2) - (stroke_width * 1))
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
    svg.append("line")
        .attr("class", "timestep-" + timestep + " " + class_suffix)
        .attr("x1", x_offset + ((size - stroke_length) / 2))
        .attr("y1", y_offset + (size / 2) + (stroke_width * 1))
        .attr("x2", x_offset + ((size - stroke_length) / 2) + stroke_length)
        .attr("y2", y_offset + (size / 2) + (stroke_width * 1))
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
}

function drawGate(timestep, x_offset, y_offset, size) {
    var stroke_width = size/50;

    //left vertical bar
    svg.append("rect")
        .attr("class", "timestep-" + timestep)
        .attr("x", x_offset)
        .attr("y", y_offset+size/8)
        .attr("width", size/10)
        .attr("height", size-size/8)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width)
        .attr("fill", "#FFF");

    //left circle
    svg.append("circle")
        .attr("class", "timestep-" + timestep)
        .attr("cx", x_offset+size*2/39)
        .attr("cy", y_offset+size/18)
        .attr("r", size/16)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width)
        .attr("fill", "#FFF");

    //right vertical bar
    svg.append("rect")
        .attr("class", "timestep-" + timestep)
        .attr("x", x_offset+size*3/2-size/8)
        .attr("y", y_offset+size/8)
        .attr("width", size/10)
        .attr("height", size-size/8)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width)
        .attr("fill", "#FFF");

    //right circle
    svg.append("circle")
        .attr("class", "timestep-" + timestep)
        .attr("cx", x_offset+size*3/2-size*2/27)
        .attr("cy", y_offset+size/18)
        .attr("r", size/16)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width)
        .attr("fill", "#FFF"); 

    //left door
    svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset+size*13/20)
        .attr("y1", y_offset)
        .attr("x2", x_offset+size*13/20)
        .attr("y2", y_offset+size-size/6)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);

   svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset+size*13/20)
        .attr("y1", y_offset)
        .attr("x2", x_offset+size/10)
        .attr("y2", y_offset+size/6)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);

   svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset+size*13/20)
        .attr("y1", y_offset+size-size/6)
        .attr("x2", x_offset+size/10)
        .attr("y2", y_offset+size-size/8)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);

   //right door
   svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset+size*4/5)
        .attr("y1", y_offset)
        .attr("x2", x_offset+size*4/5)
        .attr("y2", y_offset+size-size/6)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);

   svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset+size*4/5)
        .attr("y1", y_offset)
        .attr("x2", x_offset+size*3/2-size/8)
        .attr("y2", y_offset+size/6)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);

   svg.append("line")
        .attr("class", "timestep-" + timestep)
        .attr("x1", x_offset+size*4/5)
        .attr("y1", y_offset+size-size/6)
        .attr("x2", x_offset+size*3/2-size/8)
        .attr("y2", y_offset+size-size/8)
        .attr("stroke", black)
        .attr("stroke-width", stroke_width);
}

function drawSubTitle(title, classes) {
    svg.append("text")
        .attr("class", classes)
        .attr("x", x_margin + 10)
        .attr("y", y_margin - 2)
        .style("font-size", "16px")
        .style("fill", black)
        .text(title);
}

function drawView(view_name, view_class, viewCallback) {
    $(".component").remove();
    $(".detail").remove();
    $(".result").remove();
    $("#main_input").prop("disabled", true);
    svg.append("rect")
        .attr("id", "view-box")
        .attr("class", view_class)
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", total_width)
        .attr("height", total_height)
        .attr("stroke", "none")
        .attr("fill", "white");
    drawSubTitle(view_name, view_class);
    $("svg").height(total_height);
    drawClose(x_margin - circle_radius + 5, y_margin - circle_radius, circle_radius, view_class, function () {
        $("." + view_class).remove();
        $(".modal").remove();
        closeSequence(false);
        closeDetail();
        drawWeightsFromSequence(0);
        $("#main_input").prop("disabled", false);
    });
    viewCallback()
}

function closeDetail() {
    input_part = null;
    input_layer = null;
    compare_sequence = [];
    compare_timestep = null;
    activationsTop = null;
    activationsBottom = null;
}

function closeSequence(clear) {
    query_timestep = null;
    d3.selectAll(".predicate-highlightable")
        .attr("stroke", dark_grey);

    if (clear) {
        query = [];
        $(".predicate").remove();
        d3.select("#sequence-match-expected").text("-");
        d3.select("#execute-box").attr("width", textWidth("execute", 14) + textWidth("-", 14) + 30)
    }
}

function drawDetail() {
    $("#query-starter").css("display", "none");
    $("#query-builder").css("display", "block");

    if (debug) {
        svg.append("line")
            .attr("class", "detail")
            .attr("x1", (total_width / 2) - 0.5)
            .attr("y1", detail_margin)
            .attr("x2", (total_width / 2) - 0.5)
            .attr("y2", total_height - detail_margin)
            .attr("stroke", "blue")
            .attr("stroke-width", 1);
        svg.append("line")
            .attr("class", "detail")
            .attr("x1", (detail_margin * 2) + (((total_width / 2) - (detail_margin * 3)) / 2))
            .attr("y1", detail_margin)
            .attr("x2", (total_width / 4) + (detail_margin / 2) - 0.5)
            .attr("y2", total_height - detail_margin)
            .attr("stroke", "blue")
            .attr("stroke-width", 1);
        svg.append("line")
            .attr("class", "detail")
            .attr("x1", detail_margin)
            .attr("y1", (total_height / 2) - 0.5)
            .attr("x2", total_width - detail_margin)
            .attr("y2", (total_height / 2) - 0.5)
            .attr("stroke", "blue")
            .attr("stroke-width", 1);
    }

    drawSequenceWheel(true, main_sequence, main_timestep);
    var compareWidth = textWidth("compared to..", 14);
    svg.append("rect")
        .attr("class", "detail compare-button")
        .attr("x", (total_width / 4) + (detail_margin / 2) - (compareWidth / 2))
        .attr("y", (detail_margin * 4) + HEIGHT)
        .attr("width", compareWidth + 5)
        .attr("height", HEIGHT)
        .attr("rx", 5)
        .attr("ry", 5)
        .attr("stroke", "none")
        .attr("fill", light_grey);
    svg.append("text")
        .attr("class", "detail compare-button")
        .attr("x", (total_width / 4) + (detail_margin / 2) - (compareWidth / 2) + 2.5)
        .attr("y", (detail_margin * 4) + HEIGHT + (HEIGHT * .7))
        .style("font-size", "14px")
        .style("fill", black)
        .text("compared to..");
    svg.append("rect")
        .attr("class", "detail compare-button")
        .attr("x", (total_width / 4) + (detail_margin / 2) - (compareWidth / 2))
        .attr("y", (detail_margin * 4) + HEIGHT)
        .attr("width", compareWidth + 5)
        .attr("height", HEIGHT)
        .attr("rx", 5)
        .attr("ry", 5)
        .attr("stroke", dark_grey)
        .attr("stroke-width", 1)
        .attr("fill", "transparent")
        .style("pointer-events", "bounding-box")
        .on("mouseover", function(d) {
            d3.select(this)
                .style("cursor", "pointer");
            d3.select(this)
                .transition()
                .duration(100)
                .attr("stroke-width", 2);
        })
        .on("mouseout", function(d) {
            d3.select(this)
                .style("cursor", "auto");
            d3.select(this)
                .transition()
                .duration(50)
                .attr("stroke-width", 1);
        })
        .on("click", function(d) {
            function acceptCompareInput(sequence) {
                compare_sequence = sequence;
                $(".compare-button").remove();
                var count = activationsTop.length;
                $(".detail.load").remove();
                $(".detail.inset").remove();
                loadInset(true);
                loadDetail(true);
                drawSequenceWheel(false, compare_sequence, 0);
                drawCompareDial(count);
            }
            drawInputModal(acceptCompareInput);
        });
}

function loadInset(main) {
    var sequence = main ? main_sequence : compare_sequence;
    var timestep = main ? main_timestep : compare_timestep;
    // Load the data based off the center item.
    var slice = sequence.slice(0, timestep + 1);
    var distance = sequence.length - timestep - 1;
    console.log("Drawing inset for " + (slice.length - 1) + " @" + distance + ": " + slice);
    var placement = main ? (compare_sequence.length == 0 ? null : "top") : "bottom";
    d3.json("weights?distance=" + distance + "&" + slice.map(s => "sequence=" + encodeURI(s)).join("&"))
        .get(function (error, data) { drawInset(data, placement); });
}

function drawInset(data, placement) {
    $(".detail.inset" + (placement == null ? "" : "." + placement)).remove();
    // Shouldn't be necessary, but probably rounding errors making this look more correct.
    //                                                                               vvvvv
    var inset_height = (total_height / 2) - (state_height * 2) - (detail_margin * 4) - 0.5;
    var inset_unit_width = 15;
    var inset_unit_height = inset_height / 3;
    var inset_separator = inset_unit_width * 2.5;
    var inset_width = (inset_unit_width * 11) + (inset_separator * 8);
    var inset_x_offset = (((total_width / 2) - detail_margin) / 2) - (inset_width / 2);
    var inset_y_offset = (total_height / 2) + (state_height * 2) + (detail_margin * 2);
    if ((inset_separator * 8) + (inset_unit_width * 11) != inset_width) {
        throw (inset_separator * 8) + (inset_unit_width * 11) + " != " + inset_width;
    }
    var classes = "detail inset" + (placement == null ? "" : " " + placement);
    var label = placement == null ? null : (placement == "top" ? "A" : "B");

    if (label != null) {
        svg.append("rect")
            .attr("class", classes)
            .attr("x", inset_x_offset - (inset_unit_width * 2))
            .attr("y", inset_y_offset + (inset_height / 2) - (inset_unit_height / 2) + (placement == "bottom" ? (inset_unit_height / 2) : 0))
            .attr("width", inset_unit_width)
            .attr("height", (inset_unit_height / 2) - 1)
            .attr("stroke", "none")
            .attr("stroke-width", 1)
            .attr("fill", light_grey);
        svg.append("text")
            .attr("class", classes)
            .attr("x", inset_x_offset - (inset_unit_width * 2) + 2 - (placement == "top" ? 0.5 : 0))
            .attr("y", inset_y_offset + (inset_height / 2) - (inset_unit_height / 2) + HEIGHT - 2 + (placement == "bottom" ? (inset_unit_height / 2) : 0) + 3)
            .style("font-size", "16px")
            .style("fill", black)
            .text(label);
    }

    svg.append("rect")
        .attr("class", classes)
        .attr("x", inset_x_offset)
        .attr("y", inset_y_offset)
        .attr("width", inset_width)
        .attr("height", inset_height)
        .attr("stroke", light_grey)
        .attr("stroke-width", 1)
        .attr("fill", "none");
    // Embedding
    var x = inset_x_offset + inset_unit_width;
    var y_middle = inset_y_offset + (inset_height / 2) - (inset_unit_height / 2);
    var y_top = inset_y_offset + 10;
    var y_bottom = inset_y_offset + inset_height - 10 - inset_unit_height;
    drawInsetPart(x, y_middle, inset_unit_width, inset_unit_height, "embedding", null, data.embedding.colour, placement, classes, "embedding_hidden");
    x += inset_separator + inset_unit_width;
    drawInsetPart(x, y_top, inset_unit_width, inset_unit_height, "cell_previouses", 0, data.units["cell_previouses"][0].colour, placement, classes, "cell");
    drawInsetPart(x, y_bottom, inset_unit_width, inset_unit_height, "input_hats", 0, data.units["input_hats"][0].colour, placement, classes, "input_hat");
    x += inset_separator + inset_unit_width;
    drawInsetPart(x, y_top, inset_unit_width, inset_unit_height, "forgets", 0, data.units["forgets"][0].colour, placement, classes, "forget");
    drawInsetPart(x, y_bottom, inset_unit_width, inset_unit_height, "remembers", 0, data.units["remembers"][0].colour, placement, classes, "remember");
    x += inset_separator + inset_unit_width;
    drawInsetPart(x, y_middle, inset_unit_width, inset_unit_height, "cells", 0, data.units["cells"][0].colour, placement, classes, "cell");
    x += inset_separator + inset_unit_width;
    drawInsetPart(x, y_middle, inset_unit_width, inset_unit_height, "outputs", 0, data.units["outputs"][0].colour, placement, classes, "output");
    x += inset_separator + inset_unit_width;
    drawInsetPart(x, y_top, inset_unit_width, inset_unit_height, "cell_previouses", 1, data.units["cell_previouses"][1].colour, placement, classes, "cell");
    drawInsetPart(x, y_bottom, inset_unit_width, inset_unit_height, "input_hats", 1, data.units["input_hats"][1].colour, placement, classes, "input_hat");
    x += inset_separator + inset_unit_width;
    drawInsetPart(x, y_top, inset_unit_width, inset_unit_height, "forgets", 1, data.units["forgets"][1].colour, placement, classes, "forget");
    drawInsetPart(x, y_bottom, inset_unit_width, inset_unit_height, "remembers", 1, data.units["remembers"][1].colour, placement, classes, "remember");
    x += inset_separator + inset_unit_width;
    drawInsetPart(x, y_middle, inset_unit_width, inset_unit_height, "cells", 1, data.units["cells"][1].colour, placement, classes, "cell");
    x += inset_separator + inset_unit_width;
    drawInsetPart(x, y_middle, inset_unit_width, inset_unit_height, "outputs", 1, data.units["outputs"][1].colour, placement, classes, "output");
}

function drawInsetPart(x_offset, y_offset, width, height, part, layer, colour, placement, classes, componentName) {
    svg.append("rect")
        .attr("class", classes + " " + componentName)
        .attr("x", x_offset)
        .attr("y", placement == "bottom" ? y_offset + (height / 2) + 0.5 : y_offset)
        .attr("width", width)
        .attr("height", placement == null ? height : (height / 2) - 1)
        .attr("stroke", "none")
        .attr("stroke-width", 1)
        .attr("fill", colour);

    var part_class = "part-" + part + (layer == null ? "" : "-" + layer);
    $("." + part_class).remove();
    svg.append("rect")
        .attr("class", classes + " " + part_class + " " + componentName)
        .attr("x", x_offset)
        .attr("y", y_offset)
        .attr("width", width)
        .attr("height", height)
        .attr("stroke", input_part == part && input_layer == layer ? black : dark_grey)
        .attr("stroke-width", input_part == part && input_layer == layer ? 2 : 1)
        .attr("fill", "transparent")
        .style("pointer-events", "bounding-box")
        .on("mouseover", function(d) {
            if ((input_part != part || (input_layer != layer)) && active_components[componentName]) {
                d3.select(this)
                    .style("cursor", "pointer");
                d3.select(this)
                    .transition()
                    .duration(100)
                    .attr("stroke-width", 2);
            }

            if (active_components[componentName]) {
                $(".notation-" + componentName).parent()
                    .css("border-color", "black");
            }
        })
        .on("mouseout", function(d) {
            if ((input_part != part || (input_layer != layer)) && active_components[componentName]) {
                d3.select(this)
                    .style("cursor", "auto");
                d3.select(this)
                    .transition()
                    .duration(50)
                    .attr("stroke-width", 1);
            }

            if (active_components[componentName]) {
                $(".notation-" + componentName).parent()
                    .css("border-color", "transparent");
            }
        })
        .on("click", function(d) {
            if (active_components[componentName]) {
                input_part = part;
                input_layer = layer;
                loadInset(true);
                loadDetail(true);

                if (compare_sequence.length != 0) {
                    var count = activationsTop.length;
                    drawCompareDial(count);
                    loadInset(false);
                    loadDetail(false);
                }
            }
        });

        var flag = active_components[componentName];
        if (flag) {
            d3.selectAll("." + componentName).style("opacity", 1);
            d3.selectAll("image." + componentName).style("opacity", notationOff);
            d3.selectAll(".subtle." + componentName).style("opacity", 0.2);
        } else {
            d3.selectAll("." + componentName).style("opacity", 0);
        }
}

var activationsTop = null;
var activationsBottom = null;
var compare_dial_y_min = null;
var compare_dial_y_max = null;
var compare_dial_y_middle = null;
var compare_dial_radius = 10;
var compare_dial_similar_value = null;
var compare_dial_different_value = null;
var variance_lower_top = null;
var variance_lower_bottom = null;
var variance_upper_top = null;
var variance_upper_bottom = null;
var variance_minimum = null;
var variance_maximum = null;
var deadzone = 1;
function drawCompareDial(count) {
    var percent_width = textWidth("100%", 14);
    var match_count = count;
    var x_line = (total_width / 2) - (detail_margin * 2) - percent_width;
    compare_dial_y_min = (total_height / 2) - (total_height / 10);
    compare_dial_y_max = (total_height / 2) + (total_height / 10);
    compare_dial_y_middle = ((compare_dial_y_max - compare_dial_y_min) / 2) + compare_dial_y_min;
    compare_dial_similar_value = d3.scaleLinear()
        .domain([compare_dial_y_middle - deadzone, compare_dial_y_min + compare_dial_radius])
        .range([0, 1]);
    compare_dial_different_value = d3.scaleLinear()
        .domain([compare_dial_y_middle + deadzone, compare_dial_y_max - compare_dial_radius])
        .range([0, 1]);
    $(".detail.compare-dial").remove();
    svg.append("line")
        .attr("class", "detail compare-dial")
        .attr("x1", x_line)
        .attr("y1", compare_dial_y_min)
        .attr("x2", x_line)
        .attr("y2", compare_dial_y_max)
        .attr("stroke", black)
        .attr("stroke-width", 2);
    svg.append("text")
        .attr("class", "detail compare-dial")
        .attr("x", x_line - (textWidth("similar", 14) / 2))
        .attr("y", compare_dial_y_min - 5)
        .style("font-size", "14px")
        .style("fill", black)
        .text("similar")
    svg.append("text")
        .attr("class", "detail compare-dial")
        .attr("x", x_line - (textWidth("different", 14) / 2))
        .attr("y", compare_dial_y_max + 12)
        .style("font-size", "14px")
        .style("fill", black)
        .text("different")
    svg.append("text")
        .attr("class", "detail compare-dial compare-dial-value")
        .attr("x", x_line + compare_dial_radius + 5)
        .attr("y", compare_dial_y_middle + 5)
        .style("font-size", "14px")
        .style("fill", black)
        .text("0%")
    svg.append("text")
        .attr("class", "detail compare-dial compare-dial-count")
        .attr("x", x_line - compare_dial_radius - textWidth(match_count, 14) - 5)
        .attr("y", compare_dial_y_middle + 5)
        .style("font-size", "14px")
        .style("fill", black)
        .text(match_count)
    svg.append("circle")
        .attr("class", "detail compare-dial")
        .attr("cx", x_line)
        .attr("cy", compare_dial_y_middle)
        .attr("r", 2)
        .attr("stroke", black)
        .attr("stroke-width", 1)
        .attr("fill", black);
    svg.selectAll(".compare-dial-circle")
        .data([{}])
        .enter()
            .append("circle")
            .attr("class", "detail compare-dial compare-dial-circle")
            .attr("cx", x_line)
            .attr("cy", compare_dial_y_middle)
            .attr("r", compare_dial_radius)
            .attr("stroke", dark_grey)
            .attr("stroke-width", 1)
            .attr("fill", light_grey)
            .style("pointer-events", "bounding-box")
            .on("mouseover", function(d) {
                d3.select(this)
                    .style("cursor", "pointer");
                d3.select(this)
                    .transition()
                    .duration(100)
                    .attr("stroke-width", 2);
            })
            .on("mouseout", function(d) {
                d3.select(this)
                    .style("cursor", "auto");
                d3.select(this)
                    .transition()
                    .duration(50)
                    .attr("stroke-width", 1);
            })
           .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
}

function dragstarted(d) {
    d3.select(this)
        .raise()
        .classed("active", true);
}

function dragged(d) {
    d3.select(this)
        .attr("cy", function(d) {
            var new_y;

            if (d3.event.y < compare_dial_y_min + compare_dial_radius) {
                new_y = compare_dial_y_min + compare_dial_radius;
            } else if (d3.event.y > compare_dial_y_max - compare_dial_radius) {
                new_y = compare_dial_y_max - compare_dial_radius;
            } else {
                new_y = d3.event.y;
            }

            var percent = 0;

            if (new_y < compare_dial_y_middle - deadzone) {
                percent = Math.floor(compare_dial_similar_value(new_y) * 100);
                highlightSimilarActivations(percent / 100);
            } else if (new_y > compare_dial_y_middle + deadzone) {
                percent = Math.floor(compare_dial_different_value(new_y) * 100);
                highlightDifferentActivations(percent / 100);
            } else {
                highlightAllActivations();
            }

            $(".compare-dial-value")
                .attr("y", new_y + 5)
                .text(percent + "%");
            $(".compare-dial-count")
                .attr("y", new_y + 5);
            return new_y;
        });
}

function dragended(d) {
    d3.select(this)
        .classed("active", false);
}

function loadDetail(main) {
    var sequence = main ? main_sequence : compare_sequence;
    var timestep = main ? main_timestep : compare_timestep;
    // Load the data based off the center item.
    var slice = sequence.slice(0, timestep + 1);
    var distance = sequence.length - timestep - 1;
    console.log("Drawing detail for " + (slice.length - 1) + " @" + distance + ": " + slice);
    var layerParameter = input_layer == null ? "" : "&layer=" + input_layer;
    var placement = main ? (compare_sequence.length == 0 ? null : "top") : "bottom";
    d3.json("weight-detail?distance=" + distance + "&part=" + input_part + layerParameter + "&" + slice.map(s => "sequence=" + encodeURI(s)).join("&"))
        .get(function (error, data) { drawWeightDetail(data, placement); });
}

function drawWeightDetail(data, placement) {
    console.log(data);
    $(".detail.load" + (placement == null ? "" : "." + placement)).remove();
    var miniGeometry = {
        x: (total_width / 4) + (detail_margin / 2) - state_width,
        y: (total_height / 2) + (placement == "bottom" ? detail_margin : -((state_height * 2) + detail_margin)),
        width: state_width * 2,
        height: state_height * 2,
    };
    var fullGeometry = {
        x: (total_width / 2) + detail_margin,
        y: detail_margin * 2,
        width: (total_width / 2) - (detail_margin * 3),
        height: total_height - (detail_margin * 4),
    };
    /*if ((h / MEMORY_CHIP_HEIGHT) != (w / MEMORY_CHIP_WIDTH)) {
        throw "chips aren't square (" + (w / MEMORY_CHIP_WIDTH) + ", " + (h / MEMORY_CHIP_HEIGHT) + ")";
    }*/
    /*if (data.full.vector.length % MEMORY_CHIP_WIDTH != 0) {
        throw "vector length (" + data.full.vector.length + ") must be divisible by " + MEMORY_CHIP_WIDTH;
    }*/
    var classes = "detail load" + (placement == null ? "" : " " + placement);
    var linker_suffix = placement == null ? "-top" : "-" + placement;
    drawStateWidget(null, miniGeometry, null, data.mini.minimum, data.mini.maximum, data.mini.vector, data.mini.colour, data.mini.predictions, classes, MEMORY_CHIP_WIDTH, MEMORY_CHIP_HEIGHT, null, null, null, linker_suffix, null);
    var label = placement == null ? null : (placement == "top" ? "A:" : "B:");

    if (label != null) {
        svg.append("text")
            .attr("class", classes)
            .attr("x", miniGeometry.x - textWidth(label) - 10)
            .attr("y", miniGeometry.y + (miniGeometry.height / 2) + (HEIGHT / 4))
            .style("font-size", "16px")
            .style("fill", black)
            .text(label);
    }

    if (placement == "bottom") {
        activationsBottom = data.full.vector;
        variance_lower_bottom = data.full.minimum;
        variance_upper_bottom = data.full.maximum;
    } else {
        activationsTop = data.full.vector;
        variance_lower_top = data.full.minimum;
        variance_upper_top = data.full.maximum;
    }

    var new_variance_minimum = Math.min(variance_lower_top, variance_lower_bottom);
    var new_variance_maximum = Math.max(variance_upper_top, variance_upper_bottom);
    classes += " comparison";
    drawStateWidget(null, fullGeometry, null, new_variance_minimum, new_variance_maximum, data.full.vector, null, null, classes, DETAIL_CHIP_WIDTH, data.full.vector.length / DETAIL_CHIP_WIDTH, null, null, data.back_links, linker_suffix, placement);

    if (new_variance_minimum != variance_minimum || new_variance_maximum != variance_maximum) {
        variance_minimum = new_variance_minimum;
        variance_maximum = new_variance_maximum;

        if (placement == "top") {
            loadDetail("bottom");
        } else if (placement == "bottom") {
            loadDetail("top");
        }
    }
}

function highlightAllActivations() {
    highlightActivations(0);
}

function highlightSimilarActivations(percent) {
    highlightActivations(percent, true);
}

function highlightDifferentActivations(percent) {
    highlightActivations(percent, false);
}

function highlightActivations(percent, similar) {
    var count = 0;
    var scaler = similar ? 1.0 - percent : percent;
    var threshold = Math.max(Math.abs(variance_minimum), Math.abs(variance_maximum)) * scaler;
    var matched_opacity = similar ? 0.0 : 1.0;
    var unmatched_opacity = similar ? 1.0 : 0.0;
    var matched_counter = similar ? 0 : 1;
    var unmatched_counter = similar ? 1 : 0;
    var queryTop = [];
    var queryBottom = [];

    for (var i = 0; i < activationsTop.length; i++) {
        var activationTop = activationsTop[i];
        var activationBottom = activationsBottom[i];
        var value_top = activationTop.value;
        var value_bottom = activationBottom.value;
        var position = activationTop.position;
        if (position != activationBottom.position) {
            throw position + " != " + activationBottom.position;
        }
        var absolute_target_value = Math.max(Math.abs(value_top), Math.abs(value_bottom));
        var target_value = value_top;
        var comparison_value = value_bottom;

        if (absolute_target_value == Math.abs(value_bottom)) {
            target_value = value_bottom;
            comparison_value = value_top;
        }

        var min_matching = target_value - threshold;
        var max_matching = target_value + threshold;

        if (comparison_value < min_matching || comparison_value > max_matching) {
            setOpacities(i, matched_opacity);
            count += matched_counter;

            if (matched_counter == 1) {
                queryTop.push(position + ":" + value_top.toFixed(6));
                queryBottom.push(position + ":" + value_bottom.toFixed(6));
            }
        } else {
            setOpacities(i, unmatched_opacity);
            count += unmatched_counter;

            if (unmatched_counter == 1) {
                queryTop.push(position + ":" + value_top.toFixed(6));
                queryBottom.push(position + ":" + value_bottom.toFixed(6));
            }
        }
    }

    $(".compare-dial-count").text(count);
}

function setOpacities(index, opacity) {
    $(".comparison.activation-" + index).css("opacity", opacity);
}

var back_width_main = null;
var back_width_compare = null;
function drawSequenceWheel(main, sequence, timestep) {
    var x_offset = detail_margin * 2;
    var y_offset = (detail_margin * 3) + (main ? 0 : HEIGHT + detail_margin);
    var width = (total_width / 2) - (detail_margin * 3);
    var height = HEIGHT;
    var type_suffix = main ? "-main" : "-compare";
    $(".wheel" + type_suffix).remove();

    if (debug) {
        svg.append("rect")
            .attr("class", "detail")
            .attr("x", x_offset)
            .attr("y", y_offset)
            .attr("width", width)
            .attr("height", height)
            .attr("stroke", "blue")
            .attr("stroke-width", 1)
            .style("fill", "none");
    }

    if (main) {
        main_timestep = timestep;
    } else {
        compare_timestep = timestep;
    }

    var position = 0;
    var datums = sequence.map(word => ({position: position++, word: word}));
    svg.selectAll(".wheel")
        .data(datums)
        .enter()
            .append("text")
            .attr("id", function (d) { return "position-" + d.position + type_suffix; })
            .attr("class", "detail wheel" + type_suffix)
            .attr("x", -50)
            .attr("y", -50)
            .style("font-size", "14px")
            .style("fill", black)
            .text(function(d) { return d.word; })
            .on("mouseover", function(d) {
                if (timestep != d.position) {
                    d3.select(this)
                        .style("cursor", "pointer");
                }
            })
            .on("mouseout", function(d) {
                d3.select(this)
                    .style("cursor", "auto");
            })
            .on("click", function(d) {
                drawSequenceWheel(main, sequence, d.position);

                if (compare_sequence.length > 0) {
                    var count = activationsTop.length;
                    drawCompareDial(count);
                }
            });
    var center_item_width = textWidth(sequence[timestep], 14);
    var back_width = (center_item_width / 2);
    var front_width = (center_item_width / 2);
    $("#position-" + timestep + type_suffix)
        .attr("x", x_offset + (width / 2) - back_width)
        .attr("y", y_offset + (height / 2) + (height / 4));
    svg.append("rect")
        .attr("class", "detail wheel" + type_suffix)
        .attr("x", x_offset + (width / 2) - back_width - 2.5)
        .attr("y", y_offset)
        .attr("width", center_item_width + 5)
        .attr("height", height)
        .attr("stroke", "black")
        .attr("stroke-width", 2)
        .style("fill", "none");
    var space_width = textWidth("&nbsp;", 14) + 2;
    for (var i = 1; i <= timestep; i++) {
        var opacity = 1.0 - (Math.max(i - 3, 0) * .2);

        if (opacity > 0.0) {
            var item_width = textWidth(sequence[timestep - i], 14);
            back_width += item_width + space_width;
            $("#position-" + (timestep - i) + type_suffix)
                .attr("x", x_offset + (width / 2) - back_width)
                .attr("y", y_offset + (height / 2) + (height / 4))
                .css("opacity", opacity);
        }
    }
    for (var i = timestep + 1; i < sequence.length; i++) {
        var opacity = 1.0 - (Math.max(i - timestep - 3, 0) * .2);

        if (opacity > 0) {
            front_width += space_width;
            $("#position-" + i + type_suffix)
                .attr("x", x_offset + (width / 2) + front_width)
                .attr("y", y_offset + (height / 2) + (height / 4))
                .css("opacity", opacity);
            var item_width = textWidth(sequence[i], 14);
            front_width += item_width;
        }
    }

    if (main) {
        back_width_main = back_width;
    } else {
        back_width_compare = back_width;
    }

    if (compare_sequence.length > 0) {
        var labelA = "A:";
        var labelB = "B:";
        var item_width = textWidth(labelA, 16) + 10;
        $(".wheel-label").remove();
        svg.append("text")
            .attr("class", "detail wheel-label")
            .attr("x", x_offset + (width / 2) - Math.max(back_width_main, back_width_compare) - item_width - 2)
            .attr("y", (detail_margin * 3) + (height / 2) + (height / 4))
            .style("font-size", "16px")
            .style("fill", black)
            .text(labelA);
        svg.append("text")
            .attr("class", "detail wheel-label")
            .attr("x", x_offset + (width / 2) - Math.max(back_width_main, back_width_compare) - item_width)
            .attr("y", (detail_margin * 3) + HEIGHT + detail_margin + (height / 2) + (height / 4))
            .style("font-size", "16px")
            .style("fill", black)
            .text(labelB);
    }

    loadInset(main);
    loadDetail(main);
}

function drawOpen(x_offset, y_offset, radius, items_class, callback) {
    var id_class = "open-" + Math.random().toString(36).substring(5);
    var stroke_width = 1;
    svg.append("circle")
        .attr("class", items_class)
        .attr("cx", x_offset)
        .attr("cy", y_offset)
        .attr("r", radius)
        .attr("stroke", "none")
        .attr("stroke-width", stroke_width)
        .attr("fill", light_grey);
    svg.append("line")
        .attr("class", items_class + " " + id_class)
        .attr("x1", x_offset - (radius / 2))
        .attr("y1", y_offset)
        .attr("x2", x_offset + (radius / 2))
        .attr("y2", y_offset)
        .attr("stroke", dark_grey)
        .attr("stroke-width", stroke_width);
    svg.append("line")
        .attr("class", items_class + " " + id_class)
        .attr("x1", x_offset)
        .attr("y1", y_offset - (radius / 2))
        .attr("x2", x_offset)
        .attr("y2", y_offset + (radius / 2))
        .attr("stroke", dark_grey)
        .attr("stroke-width", stroke_width);
    svg.append("circle")
        .attr("class", items_class + " " + id_class)
        .attr("cx", x_offset)
        .attr("cy", y_offset)
        .attr("r", radius)
        .attr("stroke", dark_grey)
        .attr("stroke-width", stroke_width)
        .attr("fill", "transparent")
        .style("pointer-events", "bounding-box")
        .on("mouseover", function(d) {
            d3.select(this)
                .style("cursor", "pointer");
            d3.selectAll("." + id_class)
                .transition()
                .duration(100)
                .attr("stroke-width", stroke_width * 2);
        })
        .on("mouseout", function(d) {
            d3.select(this)
                .style("cursor", "auto");
            d3.selectAll("." + id_class)
                .transition()
                .duration(50)
                .attr("stroke-width", stroke_width);
        })
        .on("click", function(d) {
            callback();
        });
}

function drawClose(x_offset, y_offset, radius, items_class, callback) {
    var stroke_width = 1;
    var id_class = "open-" + Math.random().toString(36).substring(5);
    svg.append("circle")
        .attr("class", items_class)
        .attr("cx", x_offset)
        .attr("cy", y_offset)
        .attr("r", radius)
        .attr("stroke", "none")
        .attr("stroke-width", stroke_width)
        .attr("fill", light_grey)
    svg.append("line")
        .attr("class", items_class + " " + id_class)
        .attr("x1", x_offset - (radius / 2))
        .attr("y1", y_offset)
        .attr("x2", x_offset + (radius / 2))
        .attr("y2", y_offset)
        .attr("stroke", dark_grey)
        .attr("stroke-width", stroke_width);
    svg.append("circle")
        .attr("class", items_class + " " + id_class)
        .attr("cx", x_offset)
        .attr("cy", y_offset)
        .attr("r", radius)
        .attr("stroke", dark_grey)
        .attr("stroke-width", 1)
        .attr("fill", "transparent")
        .style("pointer-events", "bounding-box")
        .on("mouseover", function(d) {
            d3.select(this)
                .style("cursor", "pointer");
            d3.selectAll("." + id_class)
                .transition()
                .duration(100)
                .attr("stroke-width", stroke_width * 2);
        })
        .on("mouseout", function(d) {
            d3.select(this)
                .style("cursor", "auto");
            d3.selectAll("." + id_class)
                .transition()
                .duration(50)
                .attr("stroke-width", stroke_width);
        })
        .on("click", function(d) {
            callback();
        });
}

function drawAutocomplete(timestep) {
    var x_offset = x_margin;
    var y_offset = y_margin + (timestep * layer_height)
    var focus = null;

    $("#autocomplete-" + timestep).parent().remove();
    svg.append("foreignObject")
        .attr("class", "autocomplete")
        .attr("transform", "translate(" + x_offset + "," + (y_offset + state_height + (state_height / 4) - (HEIGHT / 2) - 1) + ")")
        .attr("width", input_width)
        .attr("height", HEIGHT)
        .append("xhtml:div")
        .attr("id", "autocomplete-" + timestep);
    var autocomplete = $("#autocomplete-" + timestep);
    autocomplete.append("<input class=':focus textbox'/>");
    autocomplete.find("input").focus();
    autocomplete.on("input", function() {
        autocomplete.find(".autocomplete-option").remove();
        focus = -1;
        var value = autocomplete.find("input").val();

        if (value === "") {
            return false;
        }

        for (var i = 0; i < words.length; i++) {
            if (words[i].substr(0, value.length).toLowerCase() === value.toLowerCase()) {
                autocomplete.append("<div class='autocomplete-option'>" + words[i] + "</div>");
            }
        }

        $(".autocomplete-option").on("click", function(e) {
            autocomplete.find(".autocomplete-option").remove();
            var textContent = e.target.textContent.toLowerCase();
            autocomplete.find("input").val(textContent);

            if (timestep >= main_sequence.length) {
                main_sequence.push(textContent);
                drawAutocomplete(timestep + 1);
            } else {
                main_sequence[timestep] = textContent;
            }

            drawWeightsFromSequence(timestep);
            $("#main_input").val(main_sequence.join(" "));
        });
    })
    .on("keydown", function(e) {
        var options = autocomplete.find(".autocomplete-option").length;

        // Down key
        if (e.keyCode === 40) {
            if (focus == options - 1) {
                focus = -1;
            } else {
                focus += 1;
            }
        }
        // Up key
        else if (e.keyCode === 38) {
            if (focus == -1) {
                focus = options - 1;
            } else {
                focus -= 1;
            }
        }
        // Enter key
        else if (e.keyCode == 13) {
            var selection = autocomplete.find(".autocomplete-active");

            if (selection.length == 1) {
                selection.click();
            } else {
                autocomplete.find(".autocomplete-option").remove();
                var textContent = autocomplete.find("input").val().toLowerCase();
                autocomplete.find("input").val(textContent);

                if (textContent == "") {
                    var tail = main_sequence.length;
                    main_sequence = main_sequence.slice(0, timestep);
                    trimSequenceTail(tail, main_sequence.length);
                } else {
                    if (timestep >= main_sequence.length) {
                        main_sequence.push(textContent);
                        drawAutocomplete(timestep + 1);
                    } else {
                        main_sequence[timestep] = textContent;
                    }
                }

                drawWeightsFromSequence(timestep);
                $("#main_input").val(main_sequence.join(" "));
            }
        }

        autocomplete.find(".autocomplete-active").removeClass("autocomplete-active");

        if (focus >= 0) {
            autocomplete.find(".autocomplete-option:eq(" + focus + ")").addClass("autocomplete-active");
        }
    });
}

function trimSequenceTail(old_sequence_length, new_sequence_length) {
    for (var s = new_sequence_length; s <= old_sequence_length; s++) {
        if (s != new_sequence_length) {
            $("#autocomplete-" + s).remove();
        }

        $(".timestep-" + s).remove();
    }
}

function drawWeightsFromSequence(timestep) {
    console.log("Full sequence: " + main_sequence);

    for (var s = timestep; s < main_sequence.length; s++) {
        var slice = main_sequence.slice(0, s + 1);
        var distance = main_sequence.length - s - 1;
        console.log("Drawing sequence for " + (slice.length - 1) + " @" + distance + ": " + slice);
        d3.json("weights?distance=" + distance + "&" + slice.map(s => "sequence=" + encodeURI(s)).join("&"))
            .get(function (error, data) { drawTimestep(slice.length - 1, data); });
    }
}

function drawInputModal(callback, edit_sequence) {
    var width = (total_width / 2);
    var height = (total_height / 2);
    var x_offset = (total_width - width) / 2;
    var y_offset = (total_height - height) / 2;
    svg.append("rect")
        .attr("class", "modal")
        .attr("x", x_offset)
        .attr("y", y_offset)
        .attr("width", width)
        .attr("height", height)
        .attr("stroke", black)
        .attr("stroke-width", 2)
        .attr("fill", "white");
    drawClose(x_offset, y_offset, (state_width / 4), "modal", function () {
        $(".modal").remove();
    });
    svg.append("foreignObject")
        .attr("class", "modal")
        .attr("transform", "translate(" + (x_offset + detail_margin) + "," + (y_offset + detail_margin) + ")")
        .attr("width", width - (detail_margin * 2))
        .attr("height", HEIGHT)
        .append("xhtml:div")
        .attr("id", "sequence-inputter");
    var sequenceInputter = $("#sequence-inputter");
    sequenceInputter.append("<input class=':focus textbox'/>");
    sequenceInputter.find("input").focus();

    if (edit_sequence != null) {
        sequenceInputter.find("input").val(edit_sequence.join(" "));
    }

    sequenceInputter.on("keydown", function(e) {
        // Enter key
        if (e.keyCode == 13) {
            sequence = sequenceInputter.find("input")
                .val()
                .toLowerCase()
                .split(/\s+/);
            $(".modal").remove();
            callback(sequence);
        }
    });
}

function getGeometry(timestep, part, layer) {
    var x_offset = (x_margin * 2) + input_width;
    var y_offset = y_margin + (timestep * layer_height);
    var layer_offset = layer * state_width * 14;
    var b;

    switch (part) {
        case "embedding":
            b = {x: x_offset + (state_width / 2), y: y_offset + (state_height * 3 / 4)};
            break;
        case "cell_previouses":
            b = {x: x_offset + (state_width * 8 / 2) + layer_offset, y: y_offset};
            break;
        /*case "forget_gate":
            b = {x: x_offset + (w * 3) + (w / 2) + layer_offset, y: y_offset + (h * 1 / 2) + (operator_height / 2), height: operand_height};
            break;*/
        case "forgets":
            b = {x: x_offset + (state_width * 8) - (state_width / 2) + layer_offset, y: y_offset};
            break;
        case "input_hats":
            b = {x: x_offset + (state_width * 8 / 2) + layer_offset, y: y_offset + (state_height * 3 / 2)};
            break;
        /*case "remember_gate":
            b = {x: x_offset + (w * 7) + (w / 2) + layer_offset, y: y_offset + (h * 3 / 2) + (operator_height / 2), height: operand_height};
            break;*/
        case "remembers":
            b = {x: x_offset + (state_width * 8) - (state_width / 2) + layer_offset, y: y_offset + (state_height * 3 / 2)};
            break;
        /*case "forget_hat":
            b = {x: x_offset + (w * 11) + (w / 2) + layer_offset, y: y_offset + (h * 1 / 2), height: operand_height};
            break;
        case "remember_hat":
            b = {x: x_offset + (w * 11) + (w / 2) + layer_offset, y: y_offset + (h * 2 / 2) + (operator_height / 2), height: operand_height};
            break;*/
        case "cells":
            b = {x: x_offset + (state_width * 11) + layer_offset, y: y_offset + (state_height * 3 / 4)};
            break;
        /*case "cell_hats":
            b = {x: x_offset + (state_width * 11) + (state_width / 2) + layer_offset, y: y_offset + (state_height * 3 / 4)};
            break;
        case "output_gate":
            b = {x: x_offset + (w * 15) + (w / 2) + layer_offset, y: y_offset + h + (operator_height / 2), height: operand_height};
            break;*/
        case "outputs":
            b = {x: x_offset + (state_width * 15) - (state_width / 2) + layer_offset, y: y_offset + (state_height * 3 / 4)};
            break;
        case "softmax":
            // For the 2 layers v
            b = {x: x_offset + (state_width * 19) - (state_width / 2) + layer_offset, y: y_offset + (state_height * 3 / 4)};
            break;
        default:
            return null;
    }

    return Object.assign({}, {width: state_width, height: state_height}, b);
}

function textWidth(text, fontSize) {
    var temporaryDiv = document.createElement("div");
    document.body.appendChild(temporaryDiv);
    temporaryDiv.style.fontSize = "" + fontSize + "px";
    temporaryDiv.style.position = "absolute";
    temporaryDiv.style.left = -100;
    temporaryDiv.style.top = -100;
    temporaryDiv.innerHTML = text;
    var width = temporaryDiv.clientWidth;
    document.body.removeChild(temporaryDiv);
    temporaryDiv = null;
    return width;
}

function drawSequences(data) {
    console.log(data);
    var x_offset = 20;
    var y_offset = 50;

    for (var i=0; i < query.length; i++) {
        drawPredicateChip(i, x_offset + (i * 2 * 80) + 80, y_offset, 80, 20);
    }

    function y(i) { return y_offset + 30 + (i * 20); }
    var counts = [];
    var total_count = 0;

    for (var index in data.sequence_matches) {
        drawSequence(data.sequence_matches[index], x_offset, y(index), 80, 20);
        counts.push({position: index, count: data.sequence_matches[index].count});
        total_count += data.sequence_matches[index].count;
    }

    var max = d3.max(counts, function(d) { return d.count; })
    var x = d3.scaleLinear()
        .domain([0, max])
        .range([0, 80]);

    svg.selectAll(".bar")
        .data(counts)
        .enter()
            .append("rect")
            .attr("class", "sequence")
            .attr("x", x_offset + (query.length * 2 * 80) + 80)
            .attr("y", function (d) {
                return y(d.position);
            })
            .attr("width", function(d) {
                return x(d.count);
            })
            .attr("height", 20)
            .attr("stroke", black)
            .attr("stroke-width", 1)
            .attr("fill", "none");
    svg.append("text")
        .attr("class", "sequence")
        .attr("x", x_offset + (query.length * 2 * 80) + 85)
        .attr("y", function (d) {
            return y(0) - 15;
        })
        .style("font-size", "14px")
        .style("fill", black)
        .text(total_count);
    svg.selectAll(".bar")
        .data(counts)
        .enter()
            .append("text")
            .attr("class", "sequence")
            .attr("x", x_offset + (query.length * 2 * 80) + 85)
            .attr("y", function (d) {
                return y(d.position) + 15;
            })
            .style("font-size", "14px")
            .style("fill", black)
            .text(function (d) { return d.count; });
    closeSequence(true);
    addPredicateTemplate();
}

function drawSequence(sequence_match, x_offset, y_offset, width, height) {
    for (var i=0; i <= query.length; i++) {
        if (sequence_match.elides[i]) {
            svg.append("text")
                .attr("class", "sequence")
                .attr("x", x_offset + (i * 2 * width) + (width / 2) - (textWidth("...", 14) / 2))
                .attr("y", y_offset + 15)
                .style("font-size", "14px")
                .style("fill", black)
                .text("...");
        }
    }

    for (var i=0; i < query.length; i++) {
        var word = sequence_match.words[i];
        svg.append("text")
            .attr("class", "sequence")
            .attr("x", x_offset + (i * 2 * width) + 80 + (width / 2) - (textWidth(word, 14) / 2))
            .attr("y", y_offset + 15)
            .style("font-size", "14px")
            .style("fill", black)
            .text(word);
    }
}

function nameOf(name) {
    if (name.startsWith("e_")) {
        return "embedding_hidden";
    } else if (name.startsWith("tilde_c_")) {
        return "input_hat";
    } else if (name.startsWith("s_")) {
        return "remember";
    } else if (name.startsWith("l_")) {
        return "forget";
    } else if (name.startsWith("c_")) {
        return "cell";
    } else if (name.startsWith("h_")) {
        return "output";
    } else if (name.startsWith("y_")) {
        return "softmax";
    } else {
        return null;
    }
}

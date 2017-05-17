Highcharts.chart('container', {
    chart: {
      align: 'center',
        type: 'bubble',
        plotBorderWidth: 1,
        zoomType: 'xy'
    },

    legend: {
        enabled: false
    },

    title: {
        text: 'Dynamic Function Swarms Analysis'
    },

    subtitle: {
      text: 'Tools: ptrace based tools, like "perf record -g"'
    },

    xAxis: {
        gridLineWidth: 1,
        title: {
            text: 'Timestamp (s)'
        },
        labels: {
            format: '{value}'
        }
    },

    yAxis: {
        startOnTick: false,
        endOnTick: false,
        title: {
            text: 'Function ID'
        },
        labels: {
            format: '{value}'
        },
        maxPadding: 0.2,
        plotLines: [{
            color: 'black',
            dashStyle: 'dot',
            width: 2,
            value: 0,
            label: {
                align: 'right',
                style: {
                    fontStyle: 'italic'
                },
                text: '',
                x: -10
            },
            zIndex: 3
        }]
    },

    tooltip: {
        useHTML: true,
        headerFormat: '<table>',
        pointFormat: '<tr><th colspan="2"><h3>{point.func_name}</h3></th></tr>' +
            '<tr><th>Timestamp:</th><td>{point.x}(s)</td></tr>' +
            '<tr><th>Function_ID:</th><td>{point.y}</td></tr>',
        footerFormat: '</table>',
        followPointer: true
    },

    plotOptions: {
        series: {
            dataLabels: {
                enabled: true,
                format: '{point.name}'
            }
        }
    },
    series: trace_data
});

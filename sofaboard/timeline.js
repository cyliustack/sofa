Highcharts.chart('container', {
    chart: {
        type: 'scatter',
        zoomType: 'xy'
    },
    title: {
        text: 'Time Versus Functions and Events'
    },
    subtitle: {
      text: '  '
    },
    xAxis: {
        title: {
            enabled: true,
            text: 'Time (s)'
        },
        startOnTick: true,
        endOnTick: true,
        showLastLabel: true
    },
    yAxis: {
        title: {
            text: 'Performance Metrics'
        },
        type: "logarithmic"
    },
    legend: {
        layout: 'vertical',
        align: 'left',
        verticalAlign: 'top',
        x: 0,
        y: 50,
        floating: false,
        backgroundColor: (Highcharts.theme && Highcharts.theme.legendBackgroundColor) || '#FFFFFF',
        borderWidth: 1
    },
    plotOptions: {
        scatter: {
            marker: {
                radius: 2,
                states: {
                    hover: {
                        enabled: true,
                        lineColor: 'rgb(100,100,100)'
                    }
                }
            },
            states: {
                hover: {
                    marker: {
                        enabled: false
                    }
                }
            },
            tooltip: {
                headerFormat: '<b>Category: {series.name}</b><br>',
                pointFormat: '[{point.x} s]  Y:{point.y}, Name: {point.name}'
            },
            turboThreshold: 0 

        }
    },
    series: sofa_traces 
});

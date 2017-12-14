
console.time('line');
Highcharts.chart('container', {

    chart: {
        zoomType: 'x'
    },

    boost: {
        useGPUTranslations: true
    },

    title: {
        text: 'Highcharts drawing ' +  ' points'
    },

    subtitle: {
        text: 'Using the Boost module'
    },

    tooltip: {
        valueDecimals: 2
    },

    series: [{
        data: overhead_ker,
        lineWidth: 0.5
    },
    {
        data: overhead_h2d,
        lineWidth: 0.5
    },
    {
        data: overhead_d2h,
        lineWidth: 0.5
    }
    ]

});
console.timeEnd('line');

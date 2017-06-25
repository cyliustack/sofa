#  circle_inout.inp
#
#  Draw 500 points, some inside, and some outside, the unit circle.
#
#  Choose the output device.
#
set term png medium
#
#  Name the output file.
#
set output "circle_inout.png"
#
#  Set the plot title.
#
set title "Random Points in/out of Circle"
#
#  Get grid lines.
#
set grid
#
#  Set axis labels.
#
set xlabel "<--- X --->"
set ylabel "<--- Y --->"
#
#  The following command forces X and Y axes to appear the same size.
#
set size ratio 1
#
#  Draw a unit circle, and display it BEHIND everything else.
#
set object circle center ( 0, 0 ) radius 1.0 behind
#
#  Timestamp the plot.
#
set timestamp
#
#  Plot the data using X and Y ranges [0,1],
#    using the data in 'circle_in.txt',
#    marking the data with points only (a scatter plot)
#    using line type 3 (blue)
#    and point type 4 (open square)
#  and 
#    using the data in 'circle_out.txt',
#    marking the data with points only (a scatter plot)
#    using line type 1 (red)
#    and point type 3 (star)
#
plot [0:1] [0:1] "circle_in.txt" with points lt 3 pt 4, "circle_out.txt" with points lt 1 pt 3
#
#  Terminate.
#
quit

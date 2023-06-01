rm times.txt als_playground plot.png

make all

for n in 1 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_playground
done

module load gnuplot/5.2.6-fasrc01

gnuplot <<EOF
set term png
set output 'plot.png'
set key top left
set grid
set xlabel 'Number of Threads'
set ylabel 'Wall Time'
plot "times.txt" using 0:2:xticlabel(1) w l t 'Matrix factorization'
EOF
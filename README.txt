This repository contains code and results for our project for ECE405C Winter 2026. The graphs folder contains some results from 
QAOA simulations that were run on a cluster with a time limit of 12h and 40GB of memory. The job scripts contain the files that 
ran the tests as well as the bash job scripts that were used to run the tests on the clusters. 

Solutions contain raw .txt files that were obtained from running various tests. Each file n-f.txt contains tests with n number of 
items. Every line of the file is formatted as "p, qr, cr" where p is the number of layers (or Grover iterations) and qr is the res
ult of the quantum algorithm and cr is the result of the classical algorithm. 

plot_qaoa.py is a plotting script to plot the results from /solutions into /graphs. qaoa.ipynb was used to run some initial small 
tests whereas qaoa.py was the Python file used for the larger scale tests ran on the cluster. 

qtg_solver_10.py items example is the code used for solving the 10 items knapsack problem and plotting the results. The three figures (Fig1, 2 and 3) in the graphs folder are the results of this code. 
The job scripts qtg_solver.py contain the files that ran the tests as well as the bash job scripts that were used to run the tests on the clusters. qtg and qtg_new folders contain the results of running the same instances but with the QTG circuit. 

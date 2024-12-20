# Solving the Ising Problem by Noisy Quantum Genetic Algorithms

In this repository, you'll find the code to analyze the data used in the paper and/or recreate other instances of the Ising problem and solve them.

<b> The paper is under publication for the proceedings of the IEEE Symposium Series on Computational Intelligence </b>

The repository is organized as follows:

- To plot data:
  1. Drag the .txt files (ex: "qmo_N_0.txt" or "unif_N.txt") in "Plots" folder;
  2. Open the Jupyter Notebook you're interested in;
  3. Adjust the parameters of the simulation you want to plot (# of instance and best QMO version for that instance - this will just change the line style for that values);
  4. Run the notebook;
 
- To solve instances of the Ising problem via genetic algorithms:
  1. Open init.py that you can find in the simulation folder;
  2. Uncomment the commands h_initialise or pop_initialise adjusting them with grid size and output file name desired to either generate a instance file or a population file (NOTE: if you change nInd or nPop be sure to adjust run.py and reader.py (in "Plots") accordingly);
  3. Open run.py with a Python editor;
  4. Adjust conf, popfile values. Uncomment the block relative to the operator you wanna use in the runs. Execute the code;
  This will generate the output files you can plot as written above.


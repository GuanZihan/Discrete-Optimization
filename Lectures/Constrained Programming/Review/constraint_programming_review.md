# Review of Constraint Programming
## 1. Computational paradigm
**Branching** and **Pruning**
- Branching: decompose the problem into sub-problems
- Pruning: use constraints to remove values from the
  variables' domain that cannot appear in the solution
  
![Computational Paradigm](computational%20paradigm.jpg)

**Complete Method**

**Focus on Feasibility**

## 2. Difference with Integer Programming
1. **Modeling**
   Variables in constraint programming directly model the “natural” decision variables of the problem; Integer programming, on 
   the other hand, often use of more “primitive” 0/1 variables representing simpler binary decisions
   
2. **Focus**
   In integer programming, the focus is on the objective function and pruning eliminates suboptimal solutions by computing a lower bound (in the case of
minimization problems) at every node of the search tree. In constraint programming, the focus is on the
problem constraints and the pruning of infeasible candidate solutions.
   
When advanced techniques such as probing (Savelbergh 1994) or sophisticated global constraints (e.g., viewing linear programs as global constraints) are taken into
account, the differences between constraint and integer programming systems start to blur

## 3. Tricky Skills
### 3.1 Propagation Arithmetic
### 3.2 Global Constraints
### 3.3 Redundant Constraints
### 3.4 Symmetric breaking
### 3.5 Dual Modeling
### 3.6 First-fail
### 3.7 Value/Variable labeling
### 3.8 Domain Splitting


## 4. Examples
### 4.1 8-Queen Problem

### 4.2 Map Coloring Problem

### 4.3 Car Sequencing Problem
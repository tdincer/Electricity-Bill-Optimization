### Problem Formulation

#### Sets

$I$= {$i$ | $i$ is an integer, and 0 $\leq$ i $\leq$ 96 }. Set of time chunks in a given day

$J$: {$j$ | $j$ is an integer, and 0 $\leq$ j $\leq$ 3 }. Set of cars/chargers.

#### Parameters

$t_i$: Time interval for a given chunk of the day, which is 15 min = 1/4 hr.

$Trf_i$: Customer's electricity tariff at a given 15 min chunk ($/kWh).

$BL_i$: Building load at a given 15 min chunk (kW). 

$Eff_{j}$: Efficiency of the AC/DC converter in each car.

$B_j$: Battery capacity of each car (kWh).

$E_{j}$: Energy demand of each car (kWh).

$DC$: Demand cost for the maximum power used during the day, which is given as 16 $/kW.

#### Variables

$P_{i,j}$: Load on a charger at a given 15 min chunk (kW).

#### Objective


$$
{minimum\left( Bill = \left(\sum_{i=0}^{95} \sum_{j=0}^{3} (P_{i,j} + BL_i) . Trf_i . t_i\right) + max(\sum_{j=0}^{3} P_{i,j} + BL_i) . DC \right)}
$$

#### Constraints

$$
Trf_i = 
\begin{cases}
  0.4 & if & 9AM < Time < 4PM \\
  0.1 & otherwise       \\
\end{cases}
$$

$$
Eff_{i, j} =
\begin{cases}
  0.7 & if & P_i \leq 5 \\
  0.9 & if & P_i >5 \\
\end{cases}
$$

$$
E_{j} = \sum_{i=0}^{55} P_{i, j} . Eff_{i, j} . t_i = U(15, 35)_{j} - 5
$$

$$
U(15,35)_{j} + 5 <  B_{j} = 40
$$

$$
P_{i,j} = 
\begin{cases}
  0   & if & 10AM < Time < 12PM \\
  P_{i, j}   & otherwise       \\
\end{cases}
$$



#### Boundaries

$$
P_{i,j} = 
\begin{cases}
  0   & if & 12AM < Time < 10AM \\
  [0, 7]   & otherwise       \\
\end{cases}
$$




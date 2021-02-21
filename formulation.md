### Problem Formulation

$i$: Index for each 15 min chunk of a day. There are 96 chunks of 15 min in a day. So, i goes from 0 to 95.

$t_i$: Time interval for a given chunk of the day, which is 15 min = 1/4 hr.

$j$: Car index.

$P_{i,j}$: Power used from any charger at a given 15 min chunk (kW).

$Trf_i$: Customer's electricity tariff at a given 15 min chunk ($/kWh).

$BL_i$: Building load at a given 15 min chunk (kW). 

$Eff_{j}$: Efficiency of the AC/DC converter in each car.

$B_j$: Battery capacity of each car (kWh).

$E_{j}$: Energy required for a given car (kWh).

$DC$: Demand cost for the maximum power used during the day, which is given as 16 $/kW.


$$
{minimize} \left( Bill = \sum_{i=0}^{95} \sum_{j=0}^{3} (P_{i,j} + BL_i) . Trf_i . t_i + max(\sum_{j=0}^{3} P_{i,j} + BL_i) . DC \right)
$$

<center> S.T. </center>

$$
0 < P_{i,j} < 7
$$

$$
Trf_i = 
\begin{cases}
  0.4   & if & 9AM < t_i < 4PM \\
  0.1   & otherwise       \\
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
E_{j} = \sum_{i=0}^{55} P_{i, j} . Eff_{i, j} = U(15, 35)_{j} - 5
$$

$$
E_{j} <  B_{j} = 40
$$

$$
P_{i,j} = 
\begin{cases}
  0   & if & 12AM < t_i < 10AM \\
  [0, 7]   & otherwise       \\
\end{cases}
$$


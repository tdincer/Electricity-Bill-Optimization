# Electricity-Bill-Optimization

The problem is to optimize the electricity bill of a customer with 4 electric vehicles. Each vehicle has a battery of 40 kWh. The customer has four Level-2 charging stations (one for each vehicle) with a maximum charging rate of 7kW. Assume that each charger’s charging rate could be set to any float value between 0 and 7 kW. The on-board AC-DC converter in each vehicle has an efficiency of 70% for charging rates below 5kW and 90% for charging rates above 5kW.

The chargers are located behind the utility grade meter. The figure below depicts the site configuration.

<p align="center">
  <img src=./problem_img/siteconfiguration.png width="350" alt="accessibility text">
</p>


The customer’s electricity bill is based on the total site load measured at the utility grade meter. The buildingload data for a representative day is available in buildingload.csv file with the following headers: Timestamp, BuildingLoad[kW].

The customer’s electricity tariff is described below:

- For every 15-minute interval t where 9:00 AM < t < 4:00 PM, the cost of the electricity is 0.4$/kWh.
- For every other 15-minute interval, the cost of energy is 0.1$/kWh.
- In addition to the energy costs above, the maximum 15-minute site load max(SiteLoad[kw]) within the day has a corresponding demand cost of 16$/kW. 

Assume that each vehicle departs at midnight (12:00AM) and arrives at 10:00AM with 5kWh energy left in its battery. At the time of departure, the required energy for individual vehicles follows the uniform distribution U(15, 35). By contract, 95% of all of the daily trips must be successfully completed without the need to charge the vehicles before their return to the site.

Given the problem setting above, what would be the power used in each charger for the minimum electricity bill?



### Code

The complete solution to the problem is given in the `optimizebill.py` file.

To run the code:

```python
python3 optimizebill.py
```

This code writes the results in a file named dispatch.csv with the following headers: *Timestamp, Charger1[kW], Charger2[kW], Charger3[kW], Charger4[kW]*.



### Formulation

`i`: Index for each 15 min chunk of a day. There are 96 chunks of 15 min in a day. So, i goes from 0 to 95.

`t_i`: Time interval for a given chunk of the day, which is 15 min = 1/4 hr.

`j`: Car index.

`P_{i,j}`: Power used from any charger at a given 15 min chunk (kW).

`Trf_i`: Customer's electricity tariff at a given 15 min chunk ($/kWh).

`BL_i`: Building load at a given 15 min chunk (kW). 

`Eff_{j}`: Efficiency of the AC/DC converter in each car.

`B_j`: Battery capacity of each car (kWh).

`E_{j}`: Energy required for a given car (kWh).

DC: Demand cost for the maximum power used during the day, which is given as 16 $/kW.
$$
\bold{minimize} \left( Bill = \sum_{i=0}^{95} \sum_{j=0}^{3} (P_{i,j} + BL_i) . Trf_i . t_i + max(\sum_{j=0}^{3} P_{i,j} + BL_i) . DC \right)
$$

<center> S.T. </center>

$$
0 < P_{i,j}[kWh] < 7
$$

$$
Trf_i [$/kWh] = 
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
E_{j}[kWh] = \sum_{i=0}^{55} P_{i, j} . Eff_{i, j} = U(15, 35)_{j} - 5
$$

$$
E_{j} <  B_{j} = 40  kWh
$$

$$
P_{i,j}[kW] = 
\begin{cases}
  0   & if & 12AM < t_i < 10AM \\
  0-7   & otherwise       \\
\end{cases}
$$








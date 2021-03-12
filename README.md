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



### Problem Formulation

Problem formulation is available in the [formulation.pdf](./formulation.pdf) file.



### Code

The complete solution to the problem is given in the `optimizebill.py` file. You can clone the repository and run the code from the root directory. E.g.

```python
python3 optimizebill.py
```

The code writes the results in a file named dispatch.csv with the following headers: *Timestamp, Charger1[kW], Charger2[kW], Charger3[kW], Charger4[kW]*.

![](/Users/tdincer/Desktop/eiqmobilitychallange/Electricity-Bill-Optimization/Results/Results.png)



### Requirements

Python 3.6+ is required. See also [requirements.txt](./requirements.txt) file.





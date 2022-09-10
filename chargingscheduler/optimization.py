import os
import time
import argparse
import functools
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import optimize
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.units as munits
import matplotlib.dates as mdates

root_dir = Path(__file__).parent.parent


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("Finished '%s' in %.4f secs" % (func.__name__, run_time))
        return value

    return wrapper_timer


class BillOptimizer:
    """
    A class to optimize electricity bill of a customer with N cars.
    """

    def __init__(
        self,
        ncars=4,
        charger_min=0.0,
        charger_max=7,
        trf_i1=36,
        trf_i2=64,
        car_edemand_min=15,
        car_edemand_max=35,
        car_edemand=None,
        energy_remained=5,
        buildingload=(root_dir / "data/buildingload.csv").as_posix(),
        seed=41,
        ecars=None,
        arrival_time=40,
        departure_time=96,
        demand_cost=16,
        bintohour=0.25,
        seq_power_threshold=0,
    ):
        """
        Construct all the necessary atrributes for the bill optimizer.

        Parameters
        ----------
            ncars: int
                Number of cars
            charge_min: float
                Minimum output power (kW) of the charger(s)
            charger_max: float
                Maximum output power (kW) of the charger(s)
            trf_i1: int
                Tariff condition start index
            trf_i2: int
                Tariff condition stop index
            car_edemand_min: float
                Minimum of each car's energy demand distribution (kWh)
            car_edemand_max: float
                Maximum of each car's energy demand distribution (kWh)
            car_edamand: list
                Energy demand of each car (kWh)
            energy_remained: float
                Energy remained in each car on arrival (kWh)
            ecars: np.array
                Energy demand of each car (kWh)
            arrival_time: int
                Arrival time index
            departure_time: int
                Departure time index
            demand_cost: float
                Demand cost ($)
            bintohour: float
                Time period each bin covers in hours
            buildingload: str
                Path to the building load csv file
            seed: int
                Seed
            seq_power_threshold: float
                Sequential power threshold. The consecutive time bins cannot have power change greater than this value.

        """
        self.ncars = ncars
        self.charger_min = charger_min
        self.charger_max = charger_max
        self.trf_i1 = trf_i1
        self.trf_i2 = trf_i2
        self.car_edemand_min = car_edemand_min
        self.car_edemand_max = car_edemand_max
        self.car_edemand = car_edemand
        self.energy_remained = energy_remained
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.bintohour = bintohour
        self.bins = int(24 / self.bintohour)
        self.time_onsite = self.departure_time - self.arrival_time
        self.time_offsite1 = self.arrival_time
        self.time_offsite2 = self.bins - self.departure_time
        self.demand_cost = demand_cost
        self.seq_power_threshold = seq_power_threshold

        self.buildingload = self.read_buildingload(buildingload)
        self.seed = seed

        if ecars is None:
            self.ecars = self.energy_to_be_loaded()
        else:
            self.ecars = np.asarray(ecars) - self.energy_remained
            self.ncars = len(ecars)

    def get_time(self):
        """
        Returns the timestamps from the buildingload csv file

        Returns
        -------
            np.array: array containing the Timestamp values
        """
        return self.buildingload["Timestamp"].values

    def get_buildingload(self):
        """
        Returns the buildingload from the buildingload csv file

        Returns
        -------
            np.array: float array containing the buildingload values
        """
        return self.buildingload["BuildingLoad[kW]"].values

    def energy_to_be_loaded(self):
        """
        Generate energy demand of N cars.
        Energy remained in the cars' battery on the return are subtracted from the demand.

        Returns
        -------
            np.array: float array containing the energy demand of N cars.
        """
        np.random.seed(self.seed)
        if self.car_edemand is None:
            return (
                np.random.uniform(
                    self.car_edemand_min, self.car_edemand_max, self.ncars
                )
                - self.energy_remained
            )
        else:
            return np.max(
                [
                    np.array(self.car_edemand) - self.energy_remained,
                    np.zeros(self.ncars),
                ],
                0,
            )

    def read_buildingload(self, file):
        """
        Read the buildingload.csv file

        Returns
        -------
            pd.DataFrame: dataframe containing the buildingload information.
                          Columns include 'Timestamp' and 'BuildingLoad[kW]'.
        """
        df = pd.read_csv(file)
        df["Timestamp"] = pd.to_datetime(df.Timestamp)
        return df

    def customer_tariff(self):
        """
        Conditional customer tariff for 2 two periods of the day.

        Returns
        -------
            np.array: float array containing the customer tariff.
        """
        trf = np.full(self.bins, 0.1)
        trf[self.trf_i1 : self.trf_i2] = 0.4
        return trf

    def efficiency(self, p):
        """
        Efficiency of the AC/DC convert in the cars. Cars are assumed to have the same efficiency functions.

        Returns
        -------
            np.array: float array containing the converter's efficiency
        """
        return np.where(p < 5, 0.7, 0.9)

    def adjust_decision_variable(self, p):
        """
        Appends 0s to the decision variable p to match p's shape to that of building load.

        Returns
        """
        if self.time_offsite1:
            p = np.append(np.zeros(self.time_offsite1), p)
        if self.time_offsite2:
            p = np.append(p, np.zeros(self.time_offsite2))
        return p

    def bill(self, p):
        """
        The electricity bill for a given day.

        Input
        -----
            p: np.array
                Decision variable array, power output at each time bin of all chargers. It does not include the unused
                time bins (Time bins with zero power are not included).

        Returns
        -------
            np.array: a single value array containing the electricity bill for a given day.
        """
        bl = self.get_buildingload()
        tariff = self.customer_tariff()

        p = p.reshape(-1, self.time_onsite).sum(0)  # sum over all cars.
        p = self.adjust_decision_variable(
            p
        )  # add zero-power time bins to the decision variable.
        expr1 = p + bl
        bill = (expr1 * tariff).sum() * self.bintohour + max(expr1) * self.demand_cost
        return bill

    def get_bill(self):
        """
        Returns the predicted bill for the optimized charging program.

        Returns
        -------
            np.array: a single value array containing the predicted electricity bull for a given day.
        """
        return self.bill(self.res.x)

    def make_initial_guess(self):
        """
        Generates the initial guess values for the bill optimization problem.
        TODO: Add the efficiency to the equation.

        Returns
        -------
            np.array: float array containing initial guess for the power output of all chargers.
            shape = (Ncars * TimeBin)
        """
        p0 = np.array([])
        for i in range(self.ncars):
            p = np.random.uniform(0.0, 1.0, self.time_onsite)
            p = p / p.sum() * (self.ecars[i] / self.bintohour)
            p0 = np.append(p0, p)
        return p0

    def constraint(self, x, i):
        """
        Creates the energy constraint for a single car.

        Inputs
        ------
        x: decision variables array without the 0 part.
        i: car index

        Returns
        -------
            np.array: a single value array containing the energy demand equation.
        """
        p = x[i * self.time_onsite : (i + 1) * self.time_onsite]
        peff = p * self.efficiency(p)
        return np.sum(peff) * self.bintohour - self.ecars[i]

    def constraint2(self, x, i, j):
        """
        Constrain the consecutive timebins not to vary more than a threshold value.

        Inputs
        ------
            x: decision variables array without the 0 part.
            i: car index
            j: time index
        """
        p = x[i * self.time_onsite : (i + 1) * self.time_onsite]  # Car part
        return self.seq_power_threshold - np.abs(p[j] - p[j + 1])  # Time part

    def generate_constraints(self):
        """
        Wrapper for the constraint.

        Returns
        -------
            np.array: float array containing the constraints for all cars.
        """
        constraints = ()
        # Energy demand constraint for all cars.
        for i in range(self.ncars):
            constraints = constraints + (
                {"type": "eq", "fun": partial(self.constraint, i=i)},
            )

        # Sequential power change constraint
        if self.seq_power_threshold:
            for i in range(self.ncars):
                for j in range(self.time_onsite - 1):
                    constraints = constraints + (
                        {"type": "ineq", "fun": partial(self.constraint2, i=i, j=j)},
                    )
        return constraints

    def generate_bounds(self):
        """
        Creates the bounds on the power

        Returns
        -------
            tuple: array of tuples in the form ((min_power1, max_power2), ..., (minpowerN, max_powerN))
        """
        return (((self.charger_min, self.charger_max),) * self.time_onsite) * self.ncars

    @timer
    def optimize(self, verbose=False):
        p0 = self.make_initial_guess()
        constraints = self.generate_constraints()
        bounds = self.generate_bounds()

        self.res = optimize.minimize(
            self.bill,
            x0=p0,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1e5},
        )
        if verbose:
            print(self.res["message"])
            print(r"Bill: $%f" % self.res["fun"])

    def plot_result(self, showplot=True):
        """
        Plots the power output of all chargers and the building load.
        """
        converter = mdates.ConciseDateConverter()
        munits.registry[np.datetime64] = converter

        fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)

        time = self.get_time()
        bl = self.get_buildingload()

        ps = {}
        ps["Buildingload"] = bl

        for i in range(self.ncars):
            label = "Charger-" + str(i + 1)
            p = self.res.x[i * self.time_onsite : (i + 1) * self.time_onsite]
            if self.time_offsite1:
                p = np.concatenate([np.zeros(self.time_offsite1), p])
            if self.time_offsite2:
                p = np.concatenate([p, np.zeros(self.time_offsite2)])
            ps[label] = p

        ax.stackplot(time, ps.values(), labels=ps.keys())
        ax.legend(loc="upper left", prop={"size": 6})
        plt.xlabel("Time")
        plt.ylabel("Power (kW)")
        plt.ylim(0, 35)

        fmt_half_year = mdates.HourLocator(interval=2)
        ax.xaxis.set_major_locator(fmt_half_year)

        plt.savefig("../results/results.png", format="png", dpi=300)
        if showplot:
            plt.show()

    def output2df(self):
        """
        Returns
        -------
            pd.dataframe: The optimized values of the decision variables, building load, and the timestamp.
        """
        df1 = self.buildingload["Timestamp"]

        data = self.res.x.reshape(self.ncars, -1)
        if self.time_offsite1:
            data = np.concatenate(
                [np.zeros((self.ncars, self.time_offsite1)), data], axis=1
            )
        if self.time_offsite2:
            data = np.concatenate(
                [data, np.zeros((self.ncars, self.time_offsite2))], axis=1
            )
        df2 = pd.DataFrame(
            data.T, columns=["Charger{}[kW]".format(i + 1) for i in range(self.ncars)]
        )
        df1 = pd.concat([df1, df2], axis=1)
        return df1

    def save_results(self):
        """
        Saves the optimized values of the decision variables, building load, and the timestamp into a csv file.
        """
        if not os.path.isdir("../results"):
            os.mkdir("../results")
        df1 = self.output2df()
        df1.to_csv("../results/dispatch.csv", index=False)


@timer
def main(ncars, seed, showplot):
    optimizer = BillOptimizer(ncars=ncars)
    optimizer.optimize()
    optimizer.save_results()
    optimizer.plot_result(showplot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sp",
        "--showplot",
        type=bool,
        default=False,
        help="Plot the power used by each component.",
    )
    parser.add_argument("-nc", "--ncars", type=int, default=4, help="Number of cars.")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=41,
        help="Seed to generate the power needed by cars.",
    )

    args = parser.parse_args()

    main(ncars=args.ncars, seed=args.seed, showplot=args.showplot)

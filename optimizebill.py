import time
import argparse
import functools
import numpy as np
import pandas as pd
from scipy import optimize
from functools import partial
import matplotlib.pyplot as plt


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


class electricbilloptimizer:
    def __init__(self, ncars=4, charger_min=0, charger_max=7, trf_i1=36, trf_i2=64,
                 car_energy_low=15, car_energy_high=35, remaining_energy=5, buildingload='./data/buildingload.csv',
                 seed=41, ecars=None, arrival_time=40, departure_time=96, demand_cost=16, bintohour=0.25):

        self.ncars = ncars
        self.charger_min = charger_min
        self.charger_max = charger_max
        self.trf_i1 = trf_i1
        self.trf_i2 = trf_i2
        self.car_energy_low = car_energy_low
        self.car_energy_high = car_energy_high
        self.energy_remained = remaining_energy
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.bintohour = bintohour
        self.bins = int(24 / self.bintohour)
        self.time_onsite = self.departure_time - self.arrival_time
        self.time_offsite1 = self.arrival_time
        self.time_offsite2 = self.bins - self.departure_time
        self.demand_cost = demand_cost

        self.buildingload = self.read_buildingload(buildingload)
        self.seed = seed

        if ecars is None:
            self.ecars = self.energy_to_be_loaded()
        else:
            self.ecars = ecars - self.energy_remained

    def get_time(self):
        return self.buildingload['Timestamp'].values

    def get_buildingload(self):
        return self.buildingload['BuildingLoad[kW]'].values

    def energy_to_be_loaded(self):
        np.random.seed(self.seed)
        return np.random.uniform(self.car_energy_low, self.car_energy_high, self.ncars) - self.energy_remained

    def read_buildingload(self, file):
        df = pd.read_csv(file)
        df['Timestamp'] = pd.to_datetime(df.Timestamp)
        return df

    def customer_tariff(self):
        trf = np.full(self.bins, 0.1)
        trf[self.trf_i1:self.trf_i2] = 0.4
        return trf

    def car_charging_efficiency(self, p):
        return np.where(p < 5, 0.7, 0.9)

    def bill(self, p):
        bl = self.get_buildingload()
        tariff = self.customer_tariff()

        p = p.reshape(-1, self.time_onsite).sum(0)
        if self.time_offsite1:
            p = np.append(np.zeros(self.time_offsite1), p)
        if self.time_offsite2:
            p = np.append(p, np.zeros(self.time_offsite2))

        expr1 = p.reshape(-1, self.bins).sum(0) + bl
        bill = (expr1 * tariff * self.bintohour).sum() + max(expr1) * self.demand_cost
        return bill

    def make_initial_guess(self):
        p0 = np.array([])
        for i in range(self.ncars):
            p = np.random.uniform(self.charger_min, self.charger_max, self.time_onsite)
            p = p / p.sum() * self.ecars[i]
            p0 = np.append(p0, p)
        return p0

    def constraint(self, x, i):
        return np.sum(x[i * self.time_onsite:(i + 1) * self.time_onsite] * self.car_charging_efficiency(x[i * self.time_onsite:(i + 1) * self.time_onsite])) - self.ecars[i]

    def generate_constraints(self):
        constraints = ()
        for i in range(self.ncars):
            constraints = constraints + ({'type': 'eq', 'fun': partial(self.constraint, i=i)},)
        return constraints

    def generate_bounds(self):
        return (((self.charger_min, self.charger_max),) * self.time_onsite) * self.ncars

    @timer
    def optimize(self):
        p0 = self.make_initial_guess()
        constraints = self.generate_constraints()
        bounds = self.generate_bounds()

        self.res = optimize.minimize(self.bill, x0=p0, bounds=bounds, constraints=constraints,
                                     options={'maxiter': 1e+5})
        print(self.res['message'])
        print(r"Bill: $%f" % self.res['fun'])
        return self.res

    def plot_result(self):
        time = self.get_time()
        bl = self.get_buildingload()

        plt.figure(figsize=(9, 4))
        for i in range(self.ncars):
            label = 'Car-' + str(i)

            p = self.res.x[i * self.time_onsite: (i + 1) * self.time_onsite]
            if self.time_offsite1:
                p = np.concatenate([np.zeros(self.time_offsite1), p])
            if self.time_offsite2:
                p = np.concatenate([p, np.zeros(self.time_offsite2)])
            plt.plot(time, p, label=label)
        plt.plot(time, bl, label='Buildingload')
        plt.xlabel('Time')
        plt.ylabel('Power (kW)')
        plt.ylim(0, 30)
        plt.legend(loc=0)
        plt.tight_layout()
        plt.savefig("./Results/Results.png", format="png", dpi=300)

    def save_results(self):
        df1 = self.buildingload['Timestamp']

        data = self.res.x.reshape(-1, self.ncars)
        if self.time_offsite1:
            data = np.concatenate([np.zeros((self.time_offsite1, self.ncars)), data])
        if self.time_offsite2:
            data = np.concatenate([data, np.zeros((self.time_offsite2, self.ncars))])
        df2 = pd.DataFrame(data,
                           columns=['Charger{}[kW]'.format(i) for i in range(self.ncars)])
        df1 = pd.concat([df1, df2], axis=1)
        df1.to_csv('./Results/dispatch.csv', index=False)


@timer
def main(ncars, seed, plot):
    eoptimize = electricbilloptimizer(ncars=ncars)
    eoptimize.optimize()
    eoptimize.save_results()
    if plot:
        eoptimize.plot_result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', type=bool, default=True, help='Plot the power used by each component.')
    parser.add_argument('-nc', '--ncars', type=int, default=4, help='Number of cars.')
    parser.add_argument('-s', '--seed', type=int, default=41, help='Seed to generate the power needed by cars.')

    args = parser.parse_args()

    main(ncars=args.ncars, seed=args.seed, plot=args.plot)

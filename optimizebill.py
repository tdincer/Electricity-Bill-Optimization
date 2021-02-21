import numpy as np
import pandas as pd
from scipy import optimize
from functools import partial
import matplotlib.pyplot as plt


class electricbilloptimizer:
    def __init__(self, ncars=4, charger_min=0, charger_max=7, trf_i1=36, trf_i2=64,
                 car_energy_low=25, car_energy_high=35, remaining_energy=5, buildingload='./data/buildingload.csv',
                 seed=41, ecars=None):

        self.ncars = ncars
        self.charger_min = charger_min
        self.charger_max = charger_max
        self.trf_i1 = trf_i1
        self.trf_i2 = trf_i2
        self.car_energy_low = car_energy_low
        self.car_energy_high = car_energy_high
        self.energy_remained = remaning_energy
        self.buildingload = self.read_buildingload(buildingload)
        self.seed = 41

        if ecars is None:
            self.ecars = self.energy_to_be_loaded()

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
        trf = np.full(96, 0.1)
        trf[self.trf_i1:self.trf_i2] = 0.4
        return trf

    def car_charging_efficiency(self, p):
        return np.where(p < 5, 0.7, 0.9)

    def bill(self, p):
        bl = self.get_buildingload()
        tariff = self.customer_tariff()
        expr1 = p.reshape(-1, 96).sum(0) + bl
        bill = (expr1 * tariff / 4.).sum() + max(expr1) * 16.
        return bill

    def make_initial_guess(self):
        p0 = np.array([])
        for i in range(self.ncars):
            p = np.random.uniform(self.charger_min, self.charger_max, 56)
            p = p / p.sum() * self.ecars[i]
            p0 = np.append(p0, np.append(np.zeros(40), p))
        return p0

    def constraint(self, x, i):
        return np.sum(x[i * 96:(i + 1) * 96] * self.car_charging_efficiency(x[i * 96:(i + 1) * 96])) - self.ecars[i]

    def generate_constraints(self):
        constraints = ()
        for i in range(self.ncars):
            constraints = constraints + ({'type': 'eq', 'fun': partial(self.constraint, i=i)},)
        return constraints

    def generate_bounds(self):
        return ((((0, 1e-12),) * 40 + ((self.charger_min, self.charger_max),) * 56)) * self.ncars

    def optimize(self):
        p0 = self.make_initial_guess()
        constraints = self.generate_constraints()
        bounds = self.generate_bounds()

        self.res = optimize.minimize(self.bill, x0=p0, bounds=bounds, constraints=constraints)
        print(self.res['message'])
        print(f"Bill: $%f" % self.res['fun'])
        return self.res

    def plot_result(self, save=False):
        time = self.get_time()
        bl = self.get_buildingload()

        plt.figure(figsize=(9, 4))
        for i in range(self.ncars):
            label = 'Car-' + str(i)
            plt.plot(time, self.res.x[i * 96: (i + 1) * 96], label=label)
        plt.plot(time, self.res.x.reshape(-1, 96).sum(0), label='Total')
        plt.plot(time, bl[:96], label='Buildingload')
        plt.xlabel('Time')
        plt.ylabel('Power (kW)')
        plt.ylim(0, 30)
        plt.legend(loc=0)
        plt.tight_layout()
        if save:
            plt.savefig("loadcomponents.eps", format="eps")
        else:
            plt.show()


def main():
    eoptimize = electricbilloptimizer()
    eoptimize.optimize()
    eoptimize.plot_result()


if __name__ == "__main__":
    main()

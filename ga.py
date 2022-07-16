from deap import base, algorithms
from deap import creator
from deap import tools

import algelitism
from graph_show import show_graph, show_ships

import random
import matplotlib.pyplot as plt
import numpy as np
# ввод размера стола
a,b=map(int,input().split())
# ширина
POLE_SIZE = a
# высота
H_SIZE = b
# число полимино
POLYOMINOS = 4

LENGTH_CHROM = 3 * POLYOMINOS    # длина хромосомы, подлежащей оптимизации

# константы генетического алгоритма
POPULATION_SIZE = 800   # количество индивидуумов в популяции
P_CROSSOVER = 0.9       # вероятность скрещивания
P_MUTATION = 0.2        # вероятность мутации индивидуума
MAX_GENERATIONS = 50    # максимальное количество поколений
HALL_OF_FAME_SIZE = 1
# тип полимино 0-опорное 1-L образное
type_pol = [0, 1, 1, 1, 1]
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
# расположение полимино
def randomPol(total):
    ships = []
    for n in range(total):
        ships.extend([random.randint(1, POLE_SIZE), random.randint(1, H_SIZE), random.randint(0, 3)]) # x начальное y начальное и поворот

    return creator.Individual(ships)


toolbox = base.Toolbox()
toolbox.register("randomShip", randomPol, POLYOMINOS)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.randomShip)

population = toolbox.populationCreator(n=POPULATION_SIZE)

def polsFitness(individual):
    # размеры каждого полимино
    type_ship = [[2,2], [3,2], [2,2], [2,2], [2,2]]

    inf = 1000
    P0 = np.zeros((POLE_SIZE, H_SIZE))
    P = np.ones((POLE_SIZE+6, H_SIZE+6))*inf
    P[1:POLE_SIZE+1, 1:H_SIZE+1] = P0

    th = 0
    h = np.zeros((7, 7)) * th
    ship_one = np.ones((5, 5))
    v = np.zeros((7, 3)) * th

    for *ship, t, polyo in zip(*[iter(individual)] * 3, type_ship,type_pol):
        if polyo == 0:
            # если прямоугольник
            if ship[-1] + 2 % 2 == 0:
                sh = np.copy(h[:t[0] + 2, :t[1] + 2])
                sh[1:t[0] + 1, 1:t[1] + 1] = ship_one[:t[0], :t[1]]
                P[ship[0] - 1:ship[0] + t[0] + 1, ship[1] - 1:ship[1] + t[1] + 1] += sh
            else:
                sh = np.copy(h[:t[1] + 2, :t[0] + 2])
                sh[1:t[1] + 1, 1:t[0] + 1] = ship_one[:t[1], :t[0]]
                P[ship[0] - 1:ship[0] + t[1] + 1, ship[1] - 1:ship[1] + t[0] + 1] += sh
        else:
            # если L
            sh = np.copy(h[:t[0] + 2, :t[1] + 2])
            sh[1:t[0] + 1, 1] = ship_one[:t[0], 0]
            sh[1, 2:t[1] + 1] += ship_one[0, 1:t[1]]
            if ship[-1] == 0:
                P[ship[0] - 1:ship[0] + t[0] + 1, ship[1] - 1:ship[1] + t[1] + 1] += sh
            elif ship[-1] == 1:
                sh = np.rot90(sh)
                P[ship[0] - 1:ship[0] + t[1] + 1, ship[1] - 1:ship[1] + t[0] + 1] += sh
            elif ship[-1] == 2:
                sh = np.rot90(sh)
                sh = np.rot90(sh)
                P[ship[0] - 1:ship[0] + t[0] + 1, ship[1] - 1:ship[1] + t[1] + 1] += sh
            else:
                sh = np.rot90(sh)
                sh = np.rot90(sh)
                sh = np.rot90(sh)
                P[ship[0] - 1:ship[0] + t[1] + 1, ship[1] - 1:ship[1] + t[0] + 1] += sh



    s = np.sum(P[np.bitwise_and(P > 1, P < inf)])
    s += np.sum(P[P > inf+th*4])

    return s,         # кортеж

def mutPol(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.randint(0, 1) if (i+1) % 3 == 0 else random.randint(1, POLE_SIZE) if (i+1) % 3 == 1 else random.randint(1, H_SIZE)

    return individual,


toolbox.register("evaluate", polsFitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutPol, indpb=1.0 / LENGTH_CHROM)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)


'''def show(ax):
    ax.clear()
    show_ships(ax, hof.items[0], POLE_SIZE, H_SIZE)

    plt.draw()
    plt.gcf().canvas.flush_events()


plt.ion()
fig, ax = plt.subplots()
fig.set_size_inches(6, 6)

ax.set_xlim(-2, POLE_SIZE+3)
ax.set_ylim(-2, H_SIZE+3)
'''

population, logbook = algelitism.eaSimpleElitism(population, toolbox,
                                        cxpb=P_CROSSOVER,
                                        mutpb=P_MUTATION,
                                        ngen=MAX_GENERATIONS,
                                        halloffame=hof,
                                        stats=stats,
                                        #callback=(show, (ax, )),
                                        verbose=True)

maxFitnessValues, meanFitnessValues = logbook.select("min", "avg")
if int(maxFitnessValues[-1]) == 0:
    print(True)
else:
    print(False)
best = hof.items[0]
print(best)

# plt.ioff()
# plt.show()


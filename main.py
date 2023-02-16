"""
Author: Yanlin Mi
file: inst-e.tsp inst-f.tsp

command: 
python main.py inst-e.tsp 10 100 100 1 0.9 0.05
"""

import random
from Individual import *
import sys

myStudentNum = 1202208  # R1202208
random.seed(myStudentNum)


class BasicTSP:
    def __init__(
        self, _fName, _maxIterations, _popSize, _selH, _xoverProb, _mutationRate
    ):
        """
        Parameters and general variables
        Note not all parameters are currently used, it is up to you to implement how you wish to use them and where
        """

        self.population = []
        self.matingPool = []
        self.best = None
        self.popSize = int(_popSize)
        self.genSize = None
        self.selectH = int(_selH)
        self.crossoverProb = float(_xoverProb)
        self.mutationRate = float(_mutationRate)
        self.maxIterations = int(_maxIterations)
        self.fName = _fName
        self.iteration = 0
        self.data = {}

        self.readInstance()
        self.bestInitSol = self.initPopulation()

    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, "r")
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (cid, x, y) = line.split()
            self.data[int(cid)] = (int(x), int(y))
        file.close()

    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data, [])
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        return self.best.getFitness()

    def updateBest(self, candidate):
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()

    def fitSelection(self):
        """
        Your Ternary Tournament Selection (3 'teams' in each tournament) Implementation to fill the mating pool
        """
        comp1 = [
            self.matingPool[random.randint(0, self.popSize - 1)],
            self.matingPool[random.randint(0, self.popSize - 1)],
            self.matingPool[random.randint(0, self.popSize - 1)],
        ]
        comp2 = [
            self.matingPool[random.randint(0, self.popSize - 1)],
            self.matingPool[random.randint(0, self.popSize - 1)],
            self.matingPool[random.randint(0, self.popSize - 1)],
        ]
        # tournament best fitness
        best_fitness1 = min([comp1[0][1], comp1[1][1], comp1[2][1]])
        best_fitness2 = min([comp2[0][1], comp2[1][1], comp2[2][1]])
        # find 2 winners
        for item in comp1:
            if item[1] == best_fitness1:
                partnerA = item[0]
        for item in comp2:
            if item[1] == best_fitness2:
                partnerB = item[0]
        return [partnerA, partnerB]

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[random.randint(0, self.popSize - 1)]
        indB = self.matingPool[random.randint(0, self.popSize - 1)]
        return [indA[0], indB[0]]

    def crossover(self, indA, indB):
        """
        Executes an order1 crossover and returns the genes for a new individual
        """
        if random.random() > self.crossoverProb:
            child = Individual(self.genSize, self.data,
                               random.choice([indA, indB]))
            return child
        midP = random.randint(1, self.genSize)
        p1 = indA[0:midP]
        cgenes = p1 + [i for i in indB if i not in p1]
        child = Individual(self.genSize, self.data, cgenes)
        return child

    def mutationInversion(self, ind):
        """
        Your Inversion Mutation implementation
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize - 1)
        indexB = random.randint(0, self.genSize - 1)
        # find the min index and the max index
        indexMin = min(indexA, indexB)
        indexMax = max(indexA, indexB)
        # reverse the genes between min to max
        ind.genes = (
            ind.genes[:indexMin]
            + ind.genes[indexMin:indexMax][::-1]
            + ind.genes[indexMax:]
        )

    def mutation(self, ind):
        """
        Mutate an individual by swapping two cities with certain probability (i.e., mutation rate)
        This mutator performs recipricol exchange
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize - 1)
        indexB = random.randint(0, self.genSize - 1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation.
        Note we are only storing the gene values and fitness of every
        chromosome in prev pop
        """
        self.matingPool = [[ind_i.genes, ind_i.fitness]
                           for ind_i in self.population]

    def newGeneration(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        for i in range(self.popSize):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            if self.selectH == 0:
                parent1, parent2 = self.randomSelection()  # Selection
            else:
                parent1, parent2 = self.fitSelection()  # Selection
            child = self.crossover(parent1, parent2)  # Crossover
            self.mutationInversion(child)  # Mutation
            child.computeFitness()
            self.updateBest(child)
            self.population[i] = child

    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """

        self.updateMatingPool()
        self.newGeneration()

    def search(self):
        """
        General search template.
        Iterates for a given number of steps
        """
        self.iteration = 0
        while self.iteration < self.maxIterations:
            self.GAStep()
            self.iteration += 1

        return self.best.getFitness(), self.bestInitSol, self.best.genes


if len(sys.argv) < 7:
    print("Error - Incorrect input")
    print(
        "Expecting python TSP.py [instance] [number of runs] [max iterations] [population size]",
        "[selection method] [xover prob] [mutate prob] ",
    )
    sys.exit(0)


f, inst, nRuns, nIters, pop, selH, pC, pM = sys.argv

"""
Reading in parameters, but it is up to you to implement what needs implementing
e.g. selection, crossover, mutation, etc
"""
nRuns = int(nRuns)
ga = BasicTSP(inst, nIters, pop, selH, pC, pM)
bestDist, bestInitDist, bestSol = ga.search()
avgDist, avgInitDist = bestDist, bestInitDist
for i in range(1, nRuns):
    ga = BasicTSP(inst, nIters, pop, selH, pC, pM)
    dist, distInit, sol = ga.search()
    avgDist += dist
    avgInitDist += distInit
    if dist < bestDist:
        bestDist = dist
        bestSol = sol

print("Best Solution: \t", bestSol)
print(
    "Best Distance:",
    int(bestDist),
    "\t\t Average Distance:",
    int(avgDist / nRuns),
    "\t\t Average Initial Solution:",
    int(avgInitDist / nRuns),
)

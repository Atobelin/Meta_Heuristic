import math
import numpy as np
import time
import random
import sys
from os import listdir
import matplotlib.pyplot as plt

random.seed(1202208)
np.random.seed(1202208)


class GSAT_solver:

    def __init__(self, file, _h, _wp, _maxFlips, _maxRestarts):
        self.maxFlips = _maxFlips
        self.maxRestarts = _maxRestarts
        self.flips = 0
        self.restarts = 0
        self.nVars, self.nClauses, self.clauses, self.litToClauses = -1, -1, [], {}
        self.readInstance(file)
        self.state = [0 for _ in range(self.nVars+1)]
        # unsat that would go sat
        self.makecounts = np.zeros(self.nVars+1, dtype=int)
        # sat that would go unsat
        self.breakcounts = np.zeros(self.nVars+1, dtype=int)
        self.bestObj = self.nClauses+1
        self.bestSol = [0 for _ in range(self.nVars)]
        self.wp = _wp
        self.h = _h

    def readInstance(self, fName):
        file = open(fName, 'r')
        current_clause = []
        clauseInd = 0

        for line in file:
            data = line.split()

            if len(data) == 0:
                continue
            if data[0] == 'c':
                continue
            if data[0] == 'p':
                self.nVars = int(data[2])
                self.nClauses = int(data[3])

                continue
            if data[0] == '%':
                break
            if self.nVars == -1 or self.nClauses == -1:
                print("Error, unexpected data")
                sys.exit(0)

            # now data represents a clause
            for var_i in data:
                literal = int(var_i)
                if literal == 0:
                    self.clauses.append(current_clause)
                    current_clause = []
                    clauseInd += 1
                    continue
                current_clause.append(literal)
                if literal in self.litToClauses:
                    self.litToClauses[literal].add(clauseInd)
                else:
                    self.litToClauses[literal] = set([clauseInd])

        for i in range(1, self.nVars+1):
            if i not in self.litToClauses:
                self.litToClauses[i] = set()
            if -i not in self.litToClauses:
                self.litToClauses[-i] = set()

        if self.nClauses != len(self.clauses):
            print(self.nClauses, len(self.clauses))
            print("Unexpected number of clauses in the problem")
            sys.exit(0)
        file.close()

    def generateSolution(self):
        for i in range(1, self.nVars+1):
            choice = [-1, 1]
            self.state[i] = (i * random.choice(choice))

    def initial_cost(self):
        # Compute objective value of initial solution, reset counters and recompute
        self.unsat_clauses = set()
        self.obj = self.nClauses
        self.unsat_clauses = set()
        # unsat that would go sat
        self.makecounts = np.zeros(self.nVars+1, dtype=int)
        # sat that would go unsat
        self.breakcounts = np.zeros(self.nVars+1, dtype=int)
        self.breakcounts[0] = self.nClauses+1
        num_unsat = 0
        clsInd = 0
        for clause in self.clauses:
            satLits = 0
            breakV = 0
            cStatus = False
            for lit in clause:
                if lit in self.state:
                    cStatus = True
                    satLits += 1
                    breakV = lit
                if satLits > 1:
                    break
            if satLits == 1:
                self.breakcounts[abs(breakV)] += 1
            if not cStatus:
                num_unsat += 1
                self.unsat_clauses.add(clsInd)
                for lit in clause:
                    self.makecounts[abs(lit)] += 1
            clsInd += 1
        self.obj = num_unsat
        if self.bestObj == -1:
            self.bestObj = num_unsat
            self.bestSol = self.state[1:]
        # print("Initial cost", self.obj) #,"\tTest",np.sum(self.makecounts))
        # print(self.breakcounts,self.makecounts,"\n",sep="\n")

    def flip(self, variable):
        self.flips += 1
        self.state[variable] *= -1
        self.update_counts(variable)

    # Function to update objective value and counts of variables
    # Run after flipping
    def update_counts(self, variable):
        literal = self.state[variable]
        # Decrement the makecount for clauses going unsat to sat
        # Decrement the breakcount for clauses with true literal which only had 1 satisfying literal
        # Check if flipvar is only satisfying var of clause,
        # Increment breakcount if so
        for clauseInd in self.litToClauses[literal]:
            satLits = 0
            if clauseInd in self.unsat_clauses:
                for lit in self.clauses[clauseInd]:
                    self.makecounts[abs(lit)] -= 1
                # Was unsat so only flipvar now satisfies it
                self.breakcounts[variable] += 1
            else:
                # Was sat, but not due to flipvar
                for lit in self.clauses[clauseInd]:
                    if lit in self.state:
                        satLits += 1
                        if lit != literal:
                            breaklit = lit
                if satLits == 2:
                    self.breakcounts[abs(breaklit)] -= 1
        self.unsat_clauses = self.unsat_clauses - self.litToClauses[literal]
        self.obj = len(self.unsat_clauses)
        # Sat clauses involving flipvar's previous value
        # If flipvar was only satisfying var of clause, then clause now unsat
        # Increment the makecount for clauses going sat to unsat
        # Check if only 1 satisfying literal if not, update if so
        for clauseInd in self.litToClauses[literal*(-1)]:
            satLits = 0
            cStatus = False
            for lit in self.clauses[clauseInd]:
                if lit in self.state:
                    cStatus = True
                    satLits += 1
                    breaklit = lit
            if satLits == 1:
                self.breakcounts[abs(breaklit)] += 1
            if not cStatus:
                # flipvar was only 1 satisfying it
                self.breakcounts[variable] -= 1
                self.obj += 1
                self.unsat_clauses.add(clauseInd)
                for lit in self.clauses[clauseInd]:
                    self.makecounts[abs(lit)] += 1

    def selectVar(self):
        if self.h == "greedy":
            return self.greedysolve()
        elif self.h == "greedyside":
            return self.greedysidesolve()
        if self.h == "gsat":
            return self.selectGSATvar()
        elif self.h == "gwsat":
            return self.selectGWSATvar()
        else:
            return self.selectGSATnegvar()

    def selectGWSATvar(self):
        if random.random() < self.wp:
            nextvar = self.selectRWvar()
        else:
            nextvar = self.selectGSATvar()
        return nextvar

    def selectGSATvar(self):
        gains = self.makecounts-self.breakcounts
        hvars = np.where(gains == np.amax(gains))[0]
        return np.random.choice(hvars)

    def selectRWvar(self):
        hvars = np.where(self.makecounts > 0)[0]
        return np.random.choice(hvars)

    # Fill in the code for this function
    def selectGSATnegvar(self):
        if random.random() < self.wp:
            nextvar = self.selectRWnegvar()
        else:
            nextvar = self.selectGSATvar()
        return nextvar

    def selectRWnegvar(self):
        mvars = np.where(self.makecounts > 0)[0]
        # Use a min breakcounts instead of random
        # Initialize the min gain to a high value
        min_breakcounts = math.inf
        hvar = 0
        for m in mvars:
            # If the min breakcounts is smaller than the current min breakcounts, update the min breakcounts
            if self.breakcounts[m] < min_breakcounts:
                min_breakcounts = self.breakcounts[m]
                hvar = m
        return hvar

    # Fill in greedy method here (can have greedyside as condition in this or as separate function)
    def greedysolve(self):
        # Initialize the best gain to a low value
        max_gain = -math.inf
        # Initialize the index of the best variable
        max_index = 0
        for i in range(self.nVars+1):
            # If the net gain is greater than the current best gain, update the best gain and best index
            gain = self.makecounts[i]-self.breakcounts[i]
            if gain > max_gain:
                max_gain = gain
                max_index = i
        return max_index

    def greedysidesolve(self):
        # Initialize the best gain to a low value
        max_gain = -math.inf
        # Initialize the index of the best variable
        max_index = 0
        side_list = []
        for i in range(self.nVars+1):
            gain = self.makecounts[i]-self.breakcounts[i]
            # If the net gain is greater than the current best gain, update the best gain and best index
            if gain > max_gain:
                max_gain = gain
                max_index = i
            if gain == 0:
                side_list.append(i)
        # If the net gain is zero, randomly select a variable
        if max_gain == 0:
            max_index = random.choice(side_list)
        return max_index

    def solve(self):
        # startT =  time.time()
        # self.initialize()
        self.restarts = 0
        while self.restarts < self.maxRestarts and self.bestObj > 0:
            self.restarts += 1
            self.generateSolution()
            self.initial_cost()
            self.flips = 0
            while self.flips < self.maxFlips and self.bestObj > 0:
                nextvar = self.selectVar()
                self.flip(nextvar)
                if self.obj < self.bestObj:
                    self.bestObj = self.obj
                    self.bestSol = self.state[1:]

        if self.bestObj == 0:
            #     # print("SAT")
            #     # print("Sol:\n",self.bestSol)
            solutionChecker(self.clauses, self.bestSol)
        # else:
        #     # print("Best obj", self.bestObj, "\n"*3)
        #     solutionChecker(self.clauses, self.bestSol)
        return self.flips, self.restarts, self.bestObj


def solutionChecker(clauses, sol):
    # startPython = (round(time.time() * 1000, 2))
    unsat_clause = 0
    for clause in clauses:
        cStatus = False
        for var in clause:
            if var in sol:
                cStatus = True
                break
        if not cStatus:
            unsat_clause += 1
    # stopPython = (round(time.time() * 1000,2))
    # print("t4",stopPython-startPython)
    if unsat_clause > 0:
        print("UNSAT Clauses: ", unsat_clause)
        return False
    return True


def main():
    if len(sys.argv) < 8:
        print(len(sys.argv))
        print("Error - Incorrect input")
        print("Expecting python gsat.py [fileDir] [alg] [number of runs] [max restarts]",
              "[max flips] [walk prob] [studentNum]")
        sys.exit(0)

    _, filesDir, alg, nRuns, maxRes, maxFlips, wp, sNum = sys.argv

    lastNum = sNum[-1]
    sNum, nRuns, maxRes, maxFlips, wp = int(sNum), int(
        nRuns), int(maxRes), int(maxFlips), float(wp)
    # directory = "Inst/uf75-325" #50-218"

    # Iterate through all instances in the directory that end with
    # last value of your student number

    for filename in listdir(filesDir):
        if filename.endswith(lastNum+".cnf"):
            satInst = filesDir+"/"+filename

            avgRestarts, avgFlips, avgUnsatC, avgTime, unsolved = 0, 0, 0, 0, 0

            for i in range(nRuns):
                # print("Run",i+1, end="\t")
                random.seed(sNum + i*00)
                gsat = GSAT_solver(satInst, alg, wp, maxFlips, maxRes)
                startPython = time.process_time()
                ctrFlips, ctrRestarts, ctrObj = gsat.solve()
                stopPython = time.process_time()  # (round(time.time() * 1000,2))
                avgFlips += ctrFlips
                avgRestarts += ctrRestarts
                avgUnsatC += ctrObj
                avgTime += (stopPython-startPython)
                if ctrObj > 0:
                    unsolved += 1

            print(filename, "Solved:", nRuns - unsolved, "\tAvg Obj:", avgUnsatC/nRuns,
                  "\tAvg Restarts:", avgRestarts/nRuns, "\tAvg Flips:", avgFlips/nRuns,
                  "\tAvg Time:", avgTime/nRuns)


main()

# Runtime Distribution


def build_distribution_plot(satInst):
    runtime = []
    for i in range(100):
        # print("Run",i+1, end="\t")
        random.seed(239358 + i*00)
        gsat = GSAT_solver('UF75/'+satInst, 'gwsat', 0.3, 500, 50)
        startPython = time.process_time()
        gsat.solve()
        stopPython = time.process_time()
        runtime.append(stopPython-startPython)
    plt.hist(x=runtime)
    plt.title(satInst)
    plt.xlabel("runtime")
    plt.ylabel("count")
    plt.show()


# build_distribution_plot('uf75-068.cnf')
# build_distribution_plot('uf75-058.cnf')

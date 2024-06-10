import numpy as np
import random


class Permutation:

    def __init__(self, n):
        if isinstance(n, int):
            self.perm = np.random.permutation(n).tolist()
        elif isinstance(n, Permutation):
            self.perm = list(n.perm)
        else:
            self.perm = n
        self.places = [0] * (len(self.perm))
        for i in range(len(self.perm)):
            self.places[self.perm[i]] = i

    def size(self):
        return len(self.perm)

    def __str__(self):
        return str(self.perm)

    def mutate(self):
        a, b, c, d = random.sample(range(1, self.size()), 4)
        a_place = self.places[a]
        b_place = self.places[b]
        c_place = self.places[c]
        d_place = self.places[d]
        next_a_place = (a_place + 1) % self.size()
        next_c_place = (c_place + 1) % self.size()
        next_a = self.perm[next_a_place]
        next_c = self.perm[next_c_place]
        self.perm[next_a_place], self.perm[b_place] = self.perm[b_place], self.perm[next_a_place]
        self.perm[next_c_place], self.perm[d_place] = self.perm[d_place], self.perm[next_c_place]
        self.places[b] = next_a_place
        self.places[d] = next_c_place
        self.places[next_a] = b_place
        self.places[next_c] = d_place

    def crossover(self, b: 'Permutation'): # order crossover
        mother = self.perm
        father = b.perm
        new_perm = [-1] * self.size()
        left, right = random.sample(range(1, self.size()), 2)
        left, right = min(left, right), max(left, right)
        used = set()
        for i in range(left, right + 1):
            new_perm[i] = mother[i]
            used.add(mother[i])
        j = 0
        for i in range(b.size()):
            if father[i] in used:
                continue
            while new_perm[j] != -1:
                j += 1
            new_perm[j] = father[i]
        return Permutation(new_perm)


class Population:

    def __init__(self, dist, N=100, M=150, nu=0.1):
        self.dist = dist
        self.size = len(dist)
        self.N = N
        self.M = M
        self.nu = nu
        self.population = [Permutation(self.size) for i in range(self.N)]

    def eval_permutation(self, perm: Permutation):
        arr = perm.perm
        result = 0
        for i in range(len(arr)):
            result += self.dist[arr[i]][arr[(i + 1) % len(arr)]]
        return result

    def iteration(self):
        temp_population = [Permutation(random.choice(self.population)) for i in range(self.M)]
        for i in temp_population:
            if random.random() < self.nu:
                i.mutate()

        new_perm = []
        for i in temp_population:
            if random.random() < self.nu:
                new_perm.append(i.crossover(random.choice(temp_population)))
        for i in new_perm:
            temp_population.append(i)

        result_perm = []
        for i in range(len(temp_population)):
            result_perm.append((self.eval_permutation(temp_population[i]), i))
        result_perm.sort()
        self.population = []
        for i in range(self.N):
            self.population.append(temp_population[result_perm[i][1]])
        return result_perm[0][0], temp_population[result_perm[0][1]]


if __name__ == "__main__":
    map_size = 20
    amount_of_iterations = 10000
    dist = [[random.randint(1, 100) for i in range(map_size)] for i in range(map_size)]
    pop = Population(dist)
    prev_res = 10**9
    for i in range(10000):
        res, perm = pop.iteration()
        if res < prev_res:
            prev_res = res
            print(f"Iteration №{i + 1}. New best result: {res}. For permutation:")
            print(*perm.perm)
    print("Best result overall:", prev_res)
    print("comparing with random permutations:")
    for i in range(10):
        print(pop.eval_permutation(Permutation(map_size)))

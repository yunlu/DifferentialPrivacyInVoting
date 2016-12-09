# Differential Privacy In Voting

Interesting functions:

GenRandProfile(C, NumVotes) - C: #candidates, NumVotes: number of votes to generate
- Generate a profile uniformly at random

===================================
    Random sampling method:
===================================

BordaPlusRandom(C, V, N, gamma) - C: #candidates, V: a profile, N,gamma: parametres (calculated from epsilon)
- Generate a winner according to the random sampling method (Lee, David T. "Efficient, private, and ε-strategyproof elicitation of tournament voting rules." Proceedings of the 24th International Conference on Artificial Intelligence. AAAI Press, 2015.)

BordaPlusRandomHistogram(C, V, N, gamma, NumTrials) - C: #candidates, V: a profile, N,gamma: parametres, NumTrials: number of trials
- Generate a histogram of random sampling method results based on same profile V, but using fresh randomness each time

===================================
    Laplace mechanism:
===================================
BordaLap(C, V, eps) - C: #candidates, V: a profile, eps: epsilon value (ie. epsilon-differential privacy)
- Generate a winner according to adding Laplace noise to the Borda tally

BordaLapHistogram(C, V, eps, NumTrials) - C: #candidates, V: a profile, eps: epsilon value, NumTrials: number of trials
- Generate a histogram of BordaLap results based on same profile V, but using fresh randomness each time

===================================
    Exponential mechanism:
===================================
u2Borda(C, V, r) - C: number of candidates, V: a profile, r: a candidate
- Calculates the utility of r winning for profile V (note the "2" in u2Borda is because there is another utility function considered, but version 2 (u2) is found to be better in accuracy. Version 2 is the one described in the report)

ProbList2(eps, du, C, V) - eps: epsilon value, du: sensitivity of the utility function, C: number of candidates, V: a profile
- Generate a list of probabilities, which the ith index is the probability which candidate i should win, given profile V and the utility function defined above

===================================
    Accuracy experiment
===================================
AccuracyExperiment(C, eps, NumTrials, NumVotes) - C: number of candidates, eps: epsilon value, NumTrials: number of random profiles to generate, NumVotes: size of each randomly generated profile
- We generate NumTrials profiles, each with size NumVotes. For each profile, we run each mechanism once, and give one point to mechanism M each time M produced a correct winner (all tied winners count). Accuracy for M = TotalScore/NumTrials

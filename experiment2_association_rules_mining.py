import collections
import itertools

traDatas = ['abe', 'ae', 'abc', 'ade']


class Apriori:
    # transition set
    traDatas = []
    # transition set's length
    traLen = 0
    # frequent k set, start with 1
    k = 1
    # counting the number of transition set
    traCount = {}
    # store frequent transition
    freTran = {}
    # support
    sup = 0
    # confidence
    conf = 0
    freAllTran = {}

    def __init__(self, traDatas, sup, conf):
        self.traDatas = traDatas
        self.traLen = len(traDatas)
        self.sup = sup
        self.conf = conf

    # count frequency for each element 对每个元素进行计数
    def scanFirDatas(self):
        tmpStr = ''.join(traDatas)
        self.traCount = dict(collections.Counter(tmpStr))
        return self.traCount

    # find event with higher support, and get frequent k set
    def getFreSet(self):
        self.freTran = {}
        for tra in self.traCount.keys():
            if self.traCount[tra] >= self.sup and len(tra) == self.k:
                # store frequent set
                self.freTran[tra] = self.traCount[tra]
                # store all set
                self.freAllTran[tra] = self.traCount[tra]
        # print(self.freTran)

    # compare if k - 1 elements are equal
    def cmpTwoset(self, setA, setB):
        setA = set(setA)
        setB = set(setB)
        if len(setA - setB) == 1 and len(setB - setA) == 1:
            return True
        else:
            return False

    # connecting events, Only an element is added
    def selfConn(self):
        self.traCount = {}
        # connecting event between any two event
        for item in itertools.combinations(self.freTran.keys(), 2):
            # print(item)
            # Only an element is added
            if self.cmpTwoset(item[0], item[1]):
                key = ''.join(sorted(''.join(set(item[0]).union(set(item[1])))))
                # print(key)
                if self.cutBranch(key):
                    self.traCount[key] = 0
        # print(self.traCount)

    # count support
    def scanDatas(self):
        self.k = self.k + 1
        for tra in traDatas:
            for key in self.traCount.keys():
                self.traCount[key] = self.traCount[key] + self.findChars(tra, key)

    # if subkey of the event is not frequent, return False
    def cutBranch(self, key):
        for subKey in list(itertools.combinations(key, self.k)):
            # print(subKey)
            # print(''.join(list(subKey)) not in self.freTran.keys())
            if ''.join(list(subKey)) not in self.freTran.keys():
                return False
            else:
                return True

    def findChars(self, str, chars):
        for char in list(chars):
            if char not in str:
                return False
        return 1

    def permutation(self, string, pre_str, container):
        if len(string) == 1:
            container.append(pre_str + string)

        for idx, str in enumerate(string):
            new_str = string[:idx] + string[idx + 1:]
            new_pre_str = pre_str + str

            self.permutation(new_str, new_pre_str, container)

    def genAssRule(self):
        container = []
        ruleset = set()
        for item in self.freTran.keys():
            self.permutation(item, '', container)
        for item in container:
            for i in range(1, len(item)):
                print(item[:i] + " " + item[i:])
                ruleset.add((''.join(sorted(item[:i])), ''.join(sorted(item[i:]))))
        for rule in ruleset:
            if self.calcConfi(rule[0], rule[1]) > self.conf:
                print(rule[0] + "---->>>" + rule[1])

    # computing confidence
    def calcConfi(self, first, last):
        return self.freAllTran[''.join(sorted(first + last))] / self.freAllTran[''.join(sorted(first))]

    def algorithm(self):
        # count frequency of each element
        self.scanFirDatas()
        while self.traCount != {}:
            # find event with higher support and get frequent k set
            self.getFreSet()
            # connecting events, only an element is added, Cut branch
            self.selfConn()
            # count support
            self.scanDatas()
        print(self.freAllTran)
        # print(self.freTran)
        # mining rules
        self.genAssRule()


apriori = Apriori(traDatas, 2, 0.7)
apriori.algorithm()

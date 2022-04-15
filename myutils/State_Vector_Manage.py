from numpy import sort


def processStates(state):                                                       # List of states

    def aux(st):                                                                # Longueur = puissance de 2
        s = list(st)
        if len(s) == 2:
            return {'0': s[0], '1': s[1]}
        else:
            a0 = aux(s[:len(s) // 2])
            a1 = aux(s[len(s) // 2:])
            r = {}
            for k in a0:
                r['0' + k] = a0[k]
            for k in a1:
                r['1' + k] = a1[k]
            return r

    r = []
    for i in range(len(state)):
        r.append(aux(state[i]))

    r[0] = dict(('|' + key + '>', value) for (key, value) in r[0].items())
    return r


def processDict(d):
    r = {}
    for k in d:
        im = d[k].imag
        re = d[k].real
        if abs(im) >= 0.001 or abs(re) >= 0.001:
            r.update({k: d[k]})
    return r


def printOneState(d):                                                           # get a dict as per processStates output
    start = " "
    for k in d:
        im = d[k].imag
        re = d[k].real
        if abs(im) >= 0.001 or abs(re) >= 0.001:
            print("%s % .3f + % .3fj %s" % (start, re, im, k))
            if start[0] == " ":
                start = "+"


def strOneState(d):
    r = ''
    start = " "
    for k in d:
        im = d[k].imag
        re = d[k].real
        if abs(im) >= 0.00001 or abs(re) >= 0.00001:
            if abs(im) < 0.00001:
                r += "%s % .5f %s" % (start, re, k) + '\n'
                if start[0] == " ":
                    start = "+"
            else:
                r += "%s % .5f + % .5fj %s" % (start, re, im, k) + '\n'
                if start[0] == " ":
                    start = "+"
    return r


def getDictFinalRes(result):
    try:
        return processStates([result['statevector']])[0]
    except IndexError:
        return processDict(processStates([result])[0])


def printFinalRes(result):
    printOneState(processStates([result['statevector']])[0])


def getStrFinalRes(result):
    try:
        return strOneState(processStates([result['statevector']])[0])
    except IndexError:
        return strOneState(processStates([result])[0])
    except KeyError:
        return strOneState(result)


def ket2list(ket):
    l = []
    for s in range(1, len(ket) - 1):
        if ket[s] == '1':
            l.append(len(ket) - s - 2)
    return list(sort(l))


def list2ket(l, nqbits):
    ket = ''
    for k in range(nqbits):
        if k in l:
            ket = '1' + ket
        else:
            ket = '0' + ket
    return ket


def build_ket_coeff_dict(nel, nqbits):
    right_kets = []
    for k in range(0, 2**nqbits):
        kbin = bin(k)[2:]
        count = 0
        for j in range(len(kbin)):
            if kbin[j] == '1':
                count += 1
        if count == nel:
            right_kets.append('|' + str(kbin.zfill(nqbits)) + '>')
    return {j: [] for j in right_kets}, right_kets


def build_state_kets_dict(nel, nqbits):
    wrong_index = []
    right_kets = []
    for k in range(0, 2**nqbits):
        kbin = bin(k)[2:]
        count = 0
        for j in range(len(kbin)):
            if kbin[j] == '1':
                count += 1
        if count != nel:
            wrong_index.append(k)
        else:
            right_kets.append('|' + str(kbin.zfill(nqbits)) + '>')
    return {'e' + str(k): {j: [] for j in right_kets} for k in range(2**nqbits - len(wrong_index))}, wrong_index, right_kets
